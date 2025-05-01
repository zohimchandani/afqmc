import os
import h5py
import json

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
# cudaq.set_target("nvidia", option="mqpu")
cudaq.set_target("nvidia")

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


# Get the molecular Hamiltonian and molecular data from pyscf
data_hamiltonian = get_molecular_hamiltonian(chkptfile_rohf=chkptfile_rohf,
                                             chkptfile_cas=chkptfile_cas,
                                             num_active_electrons=num_active_electrons,
                                             num_active_orbitals=num_active_orbitals,
                                             create_cudaq_ham=True,
                                             )

hamiltonian = data_hamiltonian["hamiltonian"]
pyscf_data = data_hamiltonian["scf_data"]


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


np.save('final_state_vector_' + system + '.npy', final_state_vector)

final_state_vector = np.load('final_state_vector_' + system + '.npy')

# Get AFQMC data (hamiltonian and trial wave function) in ipie format
# using the molecular data from pyscf and the final state vector from VQE
afqmc_hamiltonian, trial_wavefunction = get_afqmc_data(pyscf_data, final_state_vector)

# Initialize AFQMC
afqmc_msd = AFQMC.build(
    pyscf_data["mol"].nelec,
    afqmc_hamiltonian,
    trial_wavefunction,
    num_walkers = 200,
    num_steps_per_block = 10,
    num_blocks = 1000,
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
np.savetxt(system + '_vqe_energy.dat', vqe_energies)
np.savetxt(system + '_afqmc_energy.dat', list(qmc_data["ETotal"]))

vqe_y = vqe_energies
vqe_x = list(range(len(vqe_y)))
plt.plot(vqe_x, vqe_y, label="VQE")

afqmc_y = list(qmc_data["ETotal"])
afqmc_x = [i + vqe_x[-1] for i in list(range(len(afqmc_y)))]
plt.plot(afqmc_x, afqmc_y, label="AFQMC")

plt.xlabel("Optimization steps")
plt.ylabel("Energy [Ha]")
plt.legend()

plt.savefig('vqe+afqmc'+system+'_plot.png')