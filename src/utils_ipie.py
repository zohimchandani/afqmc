import numpy as np
import time
from pyscf import gto, scf, ao2mo, mcscf, lib
from pyscf.lib import chkfile
import h5py
import json
import os


def signature_permutation(orbital_list):
    """
    Returns the signature of the permutation in orbital_list
    """
    if len(orbital_list) == 1:
        return 1

    transposition_count = 0
    for index, element in enumerate(orbital_list):
        for next_element in orbital_list[index + 1:]:
            if element > next_element:
                transposition_count += 1

    return (-1) ** transposition_count


def get_coeff_wf(final_state_vector, n_active_electrons, thres=1e-6):
    """
    :param final_state_vector: State vector from a VQE simulation
    :param n_active_electrons: list with number of electrons in active space
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of occupied alpha, list of occupied bets
    """
    n_qubits = int(np.log2(final_state_vector.size))
    print(f"# Preparing MSD wf")
    coeff = []
    occas = []
    occbs = []
    for j, val in enumerate(final_state_vector):
        if abs(val) > thres:
            ket = np.binary_repr(j, width=n_qubits)
            alpha_ket = ket[::2]
            beta_ket = ket[1::2]
            occ_alpha = np.where([int(_) for _ in alpha_ket])[0]
            occ_beta = np.where([int(_) for _ in beta_ket])[0]
            occ_orbitals = np.append(2 * occ_alpha, 2 * occ_beta + 1)

            if (len(occ_alpha) == n_active_electrons[0]) and (len(occ_beta) == n_active_electrons[1]):
                coeff.append(signature_permutation(occ_orbitals) * val)
                occas.append(occ_alpha)
                occbs.append(occ_beta)

    coeff = np.array(coeff, dtype=complex)
    ixs = np.argsort(np.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occas = np.array(occas)[ixs]
    occbs = np.array(occbs)[ixs]
    print(f"# MSD prepared with {len(coeff)} determinants")
    return coeff, occas, occbs


def gen_ipie_input_from_pyscf(
        scf_data: dict,
        verbose: bool = True,
        chol_cut: float = 1e-5,
        ortho_ao: bool = False,
        num_frozen_core: int = 0):
    """Generate AFQMC data from PYSCF (molecular) MCSCF simulation.

    Adapted from ipie.utils.from_pyscf: returns Hamiltonian instead of writing to files.

    :param scf_data: Dictionary containing SCF data from PYSCF.
    :type scf_data: dict
    :param verbose: Flag to control verbosity of the output.
    :type verbose: bool, optional
    :param chol_cut: Cholesky decomposition cutoff.
    :type chol_cut: float, optional
    :param ortho_ao: Flag to use orthogonal atomic orbitals.
    :type ortho_ao: bool, optional
    :param num_frozen_core: Number of frozen core orbitals.
    :type num_frozen_core: int, optional

    :return: Tuple containing the Hamiltonian components.
    :rtype: tuple

    """

    from ipie.utils.from_pyscf import generate_hamiltonian as generate_afqmc_hamiltonian
    from ipie.utils.from_pyscf import copy_LPX_to_LXmn

    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    ortho_ao_mat = scf_data["X"]
    mo_coeffs = scf_data["mo_coeff"]

    if ortho_ao:
        basis_change_matrix = ortho_ao_mat
    else:
        basis_change_matrix = mo_coeffs

        if isinstance(mo_coeffs, list) or len(mo_coeffs.shape) == 3:
            if verbose:
                print(
                    "# UHF mo coefficients found and ortho-ao == False. Using"
                    " alpha mo coefficients for basis transformation."
                )
            basis_change_matrix = mo_coeffs[0]

    ham = generate_afqmc_hamiltonian(
        mol,
        mo_coeffs,
        hcore,
        basis_change_matrix,
        chol_cut=chol_cut,
        num_frozen_core=num_frozen_core,
        verbose=verbose,
    )
    ipie_ham = (ham.H1[0], copy_LPX_to_LXmn(ham.chol), ham.ecore)

    return ipie_ham


def get_afqmc_data(scf_data, final_state_vector, chol_cut=1e-5, thres_wf=1e-4, num_frozen_core=0):
    """
    Generate the AFQMC Hamiltonian and trial wavefunction from given SCF data.

    This function takes self-consistent field (SCF) data and a final state vector
    to construct the AFQMC Hamiltonian and the associated trial wavefunction. The
    process involves generating input for the Hamiltonian using the provided SCF data,
    which includes the one-electron integrals and Cholesky vectors.

    :param scf_data: A dictionary containing SCF data including molecular
                     information and the number of active electrons.
    :type scf_data: dict
    :param final_state_vector: The final state vector to compute the wavefunction.
    :type final_state_vector: numpy.ndarray
    :param chol_cut: The threshold for perfoming the Cholesky decomposition of the two body integrals
    :type chol_cut: float
    :param thres_wf: The threshold for the wave function coeffients
    :type thres_wf: float
    :param num_frozen_core: Number of frozen core orbitals.
    :type num_frozen_core: int
    :return: A tuple containing the AFQMC Hamiltonian and the trial wavefunction.
    :rtype: tuple
    """
    from ipie.hamiltonians.generic import Generic as HamGeneric
    from ipie.trial_wavefunction.particle_hole import ParticleHole, ParticleHoleNonChunked
    print("# chol decomposition of hamiltonian")
    h1e, cholesky_vectors, e0 = gen_ipie_input_from_pyscf(scf_data,
                                                          chol_cut=chol_cut,
                                                          num_frozen_core=num_frozen_core)

    molecule = scf_data["mol"]
    n_active_electrons = scf_data["num_active_electrons"]

    num_basis = cholesky_vectors.shape[1]
    num_chol = cholesky_vectors.shape[0]

    afqmc_hamiltonian = HamGeneric(
        np.array([h1e, h1e]),
        cholesky_vectors.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    wavefunction = get_coeff_wf(final_state_vector,
                                n_active_electrons=n_active_electrons,
                                thres=thres_wf
                                )

    trial_wavefunction = ParticleHole(
        wavefunction,
        molecule.nelec,
        afqmc_hamiltonian.nbasis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=False)

    trial_wavefunction.compute_trial_energy = True
    trial_wavefunction.build()
    trial_wavefunction.half_rotate(afqmc_hamiltonian)

    return afqmc_hamiltonian, trial_wavefunction


def get_molecular_hamiltonian(
        num_active_orbitals,
        num_active_electrons,
        chkptfile_rohf=None,
        chkptfile_cas=None,
        geometry=None,
        basis="sto-3g",
        spin=0,
        charge=0,
        create_cudaq_ham=False,
        verbose=0,
        label_molecule="molecule",
        dir_save_hamiltonian="./") -> dict:
    """
     Compute the molecular Hamiltonian for a given molecule using Hartree-Fock and CASCI methods.
     :param int num_active_orbitals: Number of active orbitals for the CASCI calculation.
     :param int num_active_electrons: Number of active electrons for the CASCI calculation.
     :param chkptfile_rohf: Chkfile from pyscf
     :param chkptfile_cas: Chkfile for CASCI from pyscf
     :param geometry: Atomic coordinates of the molecule in the format required by PySCF.
     :param str basis: Basis set to be used for the calculation. Default is 'cc-pVDZ'.
     :param int spin: Spin multiplicity of the molecule. Default is 0.
     :param int charge: Charge of the molecule. Default is 0.
     :param bool create_cudaq_ham: True if cuda quantum hamiltonian should be computed
     :param int verbose: Verbosity level of the calculation. Default is 0.
     :param str label_molecule: optional label for saving the hamiltonian file
     :param str dir_save_hamiltonian: optional directory name for saving the hamiltonian file

    :return: A dictionary containing the SCF data and optionally the CUDA quantum Hamiltonian.
    :rtype: dict

     """
    if chkptfile_rohf:
        with h5py.File(chkptfile_rohf, "r") as f:
            mol_bytes = f["mol"][()]
            mol = json.loads(mol_bytes.decode('utf-8'))
            # in Bohr
            geometry = mol["_atom"]
            basis = str(mol["basis"]).replace('\"', "").replace('\'', "")
            charge = int(mol["charge"])
            spin = int(mol["spin"])
            unit = mol["unit"]

    molecule = gto.M(
        atom=geometry,
        spin=spin,
        basis=basis,
        charge=charge,
        unit=unit,  # Bohr
        verbose=verbose
    )

    print('# Start Hartree-Fock computation')
    hartee_fock = scf.ROHF(molecule)
    # Run Hartree-Fock
    if chkptfile_rohf:
        dm = hartee_fock.from_chk(chkptfile_rohf)
        hartee_fock.kernel(dm)
    else:
        hartee_fock.kernel()

    hcore = scf.hf.get_hcore(molecule)
    print("# hcore.shape", len(hcore))
    s1e = molecule.intor("int1e_ovlp_sph")
    X = get_ortho_ao(s1e)

    my_casci = mcscf.CASCI(hartee_fock, num_active_orbitals, num_active_electrons)

    if num_active_orbitals > 12:
        from pyscf import dmrgscf
        omp_num_threads = int(os.getenv('OMP_NUM_THREADS', 8))
        print(f'# running with OMP_NUM_THREADS: {omp_num_threads}')
        my_casci.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-10)
        my_casci.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
        my_casci.fcisolver.threads = omp_num_threads
        my_casci.fcisolver.memory = int(mol.max_memory / 1000)  # mem in GB
        my_casci.fcisolver.conv_tol = 1e-14

    # Run CASCI with mo coefficients from file chkptfile_cas
    mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
    e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel(mo)

    print('# Start CAS computation')
    if chkptfile_cas:
        mo = chkfile.load(chkptfile_cas, 'mcscf/mo_coeff')
        e_tot, e_cas, fcivec, mo, mo_energy = my_casci.kernel(mo)
    else:
        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()
    print('# Energy CAS', e_tot)
    h1, energy_core = my_casci.get_h1eff()
    h2 = my_casci.get_h2eff()
    h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
    tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

    n_elec = [(num_active_electrons + spin) // 2,
              (num_active_electrons - spin) // 2]

    scf_data = {"mol": molecule,
                "mo_occ": my_casci.mo_occ,
                "hcore": hcore,
                "X": X,
                "mo_coeff": my_casci.mo_coeff,
                "num_active_electrons": n_elec,
                "e_cas": e_tot}

    if create_cudaq_ham:
        from src.vqe_cudaq_qnp import get_cudaq_hamiltonian
        from openfermion import jordan_wigner
        from openfermion import generate_hamiltonian

        mol_ham = generate_hamiltonian(h1, tbi, energy_core.item())
        jw_hamiltonian = jordan_wigner(mol_ham)

        if verbose:
            print("# Preparing the cudaq Hamiltonian")
        start = time.time()
        hamiltonian_cudaq, energy_core_cudaq_ham = get_cudaq_hamiltonian(jw_hamiltonian)
        end = time.time()
        if verbose:
            print("# Time for preparing the cudaq Hamiltonian:", end - start)
            print("# Total number of terms in the spin hamiltonian = ", hamiltonian_cudaq.get_term_count())

        scf_data["energy_core_cudaq_ham"] = energy_core_cudaq_ham

        data_hamiltonian = {"hamiltonian": hamiltonian_cudaq, "scf_data": scf_data}
    else:
        data_hamiltonian = {"scf_data": scf_data}

    return data_hamiltonian


def get_ortho_ao(S, LINDEP_CUTOFF=0):
    """Generate canonical orthogonalization transformation matrix.

    Parameters
    ----------
    S : :class:`np.ndarray`
        Overlap matrix.
    LINDEP_CUTOFF : float
        Linear dependency cutoff. Basis functions whose eigenvalues lie below
        this value are removed from the basis set. Should be set in accordance
        with value in pyscf (pyscf.scf.addons.remove_linear_dep_).

    Returns
    -------
    X : :class:`np.array`
        Transformation matrix.

    from ipie for avoiding conflicting imports
    """
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:, sdiag > LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag > LINDEP_CUTOFF])
    return X
