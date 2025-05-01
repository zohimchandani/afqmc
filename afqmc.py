import os
import h5py
import json

# #For GPU acceleration of ipie, sudo -S apt-get install -y cuda-toolkit-12.4, unset CUDA_HOME and unset CUDA_PATH
from ipie.config import config
config.update_option("use_gpu", True)

from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable

import numpy as np
np.random.seed(1)

from src.utils_ipie import get_molecular_hamiltonian
from src.utils_ipie import get_afqmc_data
from src.vqe_cudaq_qnp import VQE

import matplotlib.pyplot as plt

import cudaq 
cudaq.set_target("nvidia")




# %% [markdown]
# We start by defining the structure of the molecule, the basis set, and its spin. We build the molecule object with PySCF and run a preliminary Hartree-Fock computation. Here we choose two examples: 
# 
# 1. ozone that plays a crucial role in atmospheric chemistry and environmental science and represents a challenge for single reference electronic structure methods. 
# 
# 2. a [chelating agent](https://doi.org/10.1021/acs.jctc.3c01375) representing a relevant class of substances industrially produced at large scales. Their use ranges, among the others, from water softeners in cleaning applications, modulators of redox behaviour in oxidative bleaching, scale suppressants, soil remediation and ligands for catalysts. In particular we focus here in a Fe(III)-NTA complex whose structure is given in the file imported below. We can choose two active spaces of 10 and 24 qubits.
# This simulation will require more computational resources.
# 
# 
# 

# %%
# system = 'o3' 
system = '10q' 
# system = '24q' 

if system == 'o3':

    num_active_orbitals = 6
    num_active_electrons = 8
    spin = 0
    geometry = "systems/geo_o3.xyz"
    basis = "sto-3g"
    # basis = "cc-pVDZ"
    
elif system == '10q':

    num_active_orbitals = 5
    num_active_electrons = 5
    spin = 1
    chkptfile_rohf = "chkfiles/scf_fenta_sd_converged.chk"
    chkptfile_cas = "chkfiles/10q/mcscf_fenta_converged_10q.chk"
    num_vqe_layers = 10

elif system == '24q':
    
    num_active_orbitals = 12
    num_active_electrons = 9
    spin = 1
    chkptfile_rohf = "chkfiles/scf_fenta_sd_converged.chk"
    chkptfile_cas = "chkfiles/24q/mcscf_fenta_converged_24q.chk"
    num_vqe_layers = 10
    
n_qubits = 2 * num_active_orbitals

# %% [markdown]
# ### Hamiltonian preparation for VQE
# 
# Since this molecule contains of around 600 orbitals (which would correspond to 1200 qubits) and 143 total electrons, it is impossible to perform a full VQE with full statevector simulation. Therefore, we need to identify an active space with fewer orbitals and electrons that contribute to the strongly interacting part of the whole molecule. We then run a post Hartree-Fock computation with the PySCF's built-in CASCI method in order to obtain the one-body ($t_{pq}$) and two-body 
# ($V_{prqs}$) integrals that define the molecular Hamiltonian in the active space:
# 
# $$ H= \sum_{pq}t_{pq}\hat{a}_{p}^\dagger \hat {a}_{q}+\sum_{pqrs}  V_{prqs}\hat a_{p}^\dagger \hat a_{q}^\dagger \hat a_{s}\hat a_{r} \tag{1}$$
# 

# %%

# Get the molecular Hamiltonian and molecular data from pyscf
data_hamiltonian = get_molecular_hamiltonian(chkptfile_rohf=chkptfile_rohf,
                                             chkptfile_cas=chkptfile_cas,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=True,
                                             )

hamiltonian = data_hamiltonian["hamiltonian"]
pyscf_data = data_hamiltonian["scf_data"]

# %% [markdown]
# ### Run VQE with CUDA-Q
# 

# %% [markdown]
# We can now execute the VQE algorithm using the quantum number preserving ansatz. At the end of the VQE, we store the final statevector that will be used in the classical AFQMC computation as an initial guess.
# 

# %%

# Define optimization methods for VQE
optimizer_type = 'COBYLA'

# Define options for the VQE algorithm
options = {'n_vqe_layers': num_vqe_layers,
           'maxiter': 750,
           'energy_core': pyscf_data["energy_core_cudaq_ham"],
           'return_final_state_vec': True,
           'optimizer': optimizer_type,
           'target': 'nvidia'}


# Initialize the VQE algorithm
vqe = VQE(n_qubits=n_qubits,
          num_active_electrons=num_active_electrons,
          spin=spin,
          options=options)

# Set initial parameters for the VQE algorithm
vqe.options['initial_parameters'] = np.random.rand(vqe.num_params)

# Execute the VQE algorithm
result = vqe.execute(hamiltonian)


# Extract results from the VQE execution
optimized_energy = result['energy_optimized']
vqe_energies = result["callback_energies"]
final_state_vector = result["state_vec"]
best_parameters = result["best_parameters"]


# np.save('final_state_vector_' + system + '.npy', final_state_vector)

# final_state_vector = np.load('final_state_vector_' + system + '.npy')

# %% [markdown]
# ### Auxiliary Field Quantum Monte Carlo (AFQMC)
# 

# %% [markdown]
# AFQMC is a numerical method for computing relevant properties of strongly interacting molecules. AFQMC is a type of Quantum Monte Carlo method that combines the use of random walks with an auxiliary field to simulate the imaginary-time evolution of a quantum system and drive it to the lowest energy state. This method can provide accurate results for ground-state properties of a wide range of physical systems, including atoms, molecules, and solids. Here we summarize the main features of AFQMC while a detailed introduction can be found [here](https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf).
# 

# %% [markdown]
# We consider the electronic Hamiltonian in the second quantization
# \begin{equation}
# H = {H}_1 + {H}_2 
# =\sum_{pq} h_{pq} {a}_{p}^{\dagger} {a}_{q} + \frac{1}{2} \sum_{pqrs} v_{pqrs}{a}_{p}^{\dagger} {a}_r {a}^{\dagger}_{q} {a}_s \tag{2}
# \end{equation}
# where ${a}_{p}^{\dagger}$ and ${a}_{q}$ are fermionic creation and annihilation operators of orbitals $p$ and $q$, respectively. The terms $h_{pq} $ and 
# $v_{pqrs}$ are the matrix elements of the one-body, $H_1$, and two-body, $H_2$, interactions of $H$, respectively. Here, we omit the spin indices for simplicity.
# 
# AFQMC realizes an imaginary time propagation of an initial state (chosen as a Slater determinant) $\ket{\Psi_{I}}$ towards the ground state $\ket{\Psi_0}$ of a given hamiltonian, $H$, with
# \begin{equation}
# \ket{\Psi_0} \sim\lim_{n \to \infty} \left[ e^{-\Delta\tau H  }  \right]^{n} \ket{\Psi_{I}}
# \tag{3}
# \end{equation} 
# where $\Delta\tau$ is the imaginary time step.
# 
# AFQMC relies on decomposing the two-body interactions $H_2$ in terms of sum of squares of one-body operators ${v}_\gamma$ such that the Hamiltonian ${H}$ becomes
# \begin{equation}
# H = v_0 - \frac{1}{2}\sum_{\gamma=1}^{N_\gamma} {v}_\gamma^2
# \tag{4}
# \end{equation}
# with ${v}_0 = {H}_1 $ and $
# {v}_\gamma = i \sum_{pq} L^{\gamma}_{pq} {a}_{p}^{\dagger}{a}_{q}.
# $
# The $N_\gamma$ matrices $L^{\gamma}_{pq}$ are called Cholesky vectors as they are  obtained via a Cholesky decomposition of the two-body matrix elements 
# $v_{pqrs}$ via $v_{pqrs} = \sum_{\gamma=1}^{N_\gamma} L^{\gamma}_{pr} L^{\gamma}_{qs}$.
# 
# The imaginary time propagation evolves an ensemble of walkers $\{\phi^{(n)}\}$ (that are Slater determinants) and allows one to access observables of the system. For example, the local energy
# \begin{equation}
# \mathcal{E}_{\text{loc}}(\phi^{(n)}) = \frac{\bra{\Psi_\mathrm{T}}H\ket{\phi^{(n)}}}{\braket{\Psi_\mathrm{T}| \phi^{(n)}}}
# \tag{5}
# \end{equation}
# defined as the mixed expectation value of the Hamiltonian with the trial wave function $\ket{\Psi_\mathrm{T}}$.
# 
# 
# The trial wavefunction can be in general a single or a multi-Slater determinant coming from VQE for example. This might help in achieving more accurate ground state energy estimates.
# 

# %% [markdown]
# 
# The implementation of AFQMC we use here is from [ipie](https://github.com/JoonhoLee-Group/ipie) that supports both CPUs and GPUs and requires the following steps:
# 
# 
# 1. Preparation of the molecular Hamiltonian by performing the Cholesky decomposition
# 
# 2. Preparation of the trial state from the VQE wavefunction
# 
# 3. Executing AFQMC
# 

# %% [markdown]
# ### Preparation of the molecular Hamiltonian and of the trial wave function
# 

# %%
# Get AFQMC data (hamiltonian and trial wave function) in ipie format
# using the molecular data from pyscf and the final state vector from VQE
afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

# %% [markdown]
# ### Setup of the AFQMC parameters
# 

# %% [markdown]
# Here we can choose the input options like the timestep $\Delta\tau$, the total number of walkers `num_walkers` and the total number of AFQMC iterations `num_blocks`.
# 

# %%
# Initialize AFQMC
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers = 200,
    num_steps_per_block = 10,
    num_blocks = 250,
    timestep = 0.005,
    stabilize_freq = 5,
    seed=1,
    pop_control_freq = 5,
    verbose=True)


# Run the AFQMC simulation and save data to .h5 file
afqmc_msd.run(estimator_filename='afqmc_data_' +system+ '.h5')

afqmc_msd.finalise(verbose=False)

# Extract and plot results
qmc_data = extract_observable(afqmc_msd.estimators.filename, "energy")
# np.savetxt(system + '_vqe_energy.dat', vqe_energies)
# np.savetxt(system + '_afqmc_energy.dat', list(qmc_data["ETotal"]))


# %%
vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
plt.plot(afqmc_x, afqmc_y, label="AFQMC")

plt.xlabel("Optimizaion steps")
plt.ylabel("Energy [Ha]")
plt.legend()

plt.savefig('vqe+afqmc'+system+'_plot.png')

# %% [markdown]
# If you were to pick `system == 10q` or `24q`, representing the two active spaces of FeNTA, it would produce the following plots: 
# 
# ![plot1](10+24q.png)
# 


