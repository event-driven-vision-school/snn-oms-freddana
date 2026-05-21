"""
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz
Sarka Liskova, sarka.liskova@fel.cvut.cz
Paolo Ritirato, paolo.ritirato@fel.cvut.cz

Task 1: Adding connections
The previous network consisted of independent neurons with spontaneous spiking.
Now we will add connections and make only one of the neurons spike spontaneously.
We will form a chain of 10 neurons, where signal can travel in both directions
and just the middle neuron will have non-zero baseline potential.

Run the provided script and see how the signal spreads from the spiking neuron.
Play around with the tau parameter to see how it influences the system.

Then return tau back to tau = 10*ms, disable the right-to-left connections
(S.connect(condition='j == i-1 and i > 0')), and set the non-zero baseline
potential for the first neuron instead of the middle one.
Run the simulation. How far does the signal travel? Which neuron in the chain
is the most distant one to spike?

Task 2: Connection strength
Now play with the connection strength weight w, explore values (0.9 - 1.1).
How does the spiking behaviour change?

Task 3: Refractory period
Set weights back to w = 1.0 and start decreasing the refractory period from
5*ms down to 1*ms. How far does the signal spread for 1*ms refractory period?
"""

from brian2 import *  # get by `pip install brian2`
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt


def animate_signal_transfer(VM, N):
    plt.close('all')
    n_frames = VM.v.shape[1]
    skip = 5

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.show(block=False)  # Open the window once, non-blocking

    for frame in range(0, n_frames, skip):
        ax.clear()
        v_now = VM.v[:, frame]
        ax.bar(range(N), np.clip(v_now, 0, 1.5), color='salmon')
        ax.axhline(1.0, color='k', linestyle='--', label='threshold')
        ax.set_ylim(0, 1.5)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Membrane potential')
        ax.set_title(f't = {VM.t[frame]/ms:.1f} ms')
        ax.legend()
        plt.pause(0.01)


start_scope()  # Start a new Brian2 simulation scope to reset any previous settings

N = 10          # Total number of neurons
tau = 10 * ms   # Time constant determining how quickly the membrane potential responds
duration = 120 * ms  # Total duration of the simulation

# Define the differential equation governing the neuron membrane potential dynamics
eqs = '''
dv/dt = (v0 - v) / tau : 1 (unless refractory)
v0 : 1
'''

# Create a group of N neurons
G = NeuronGroup(N, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')

# Monitors
M = SpikeMonitor(G)           # Record spike times and neuron indices
VM = StateMonitor(G, 'v', record=True)  # Record membrane potential for all neurons

# Initialize all baseline potentials to 0 (no spontaneous spiking)
G.v0 = 0

# Set the middle neuron (index 4) to have v0 > 1 so it spikes spontaneously
# and propagates its activity in both directions along the chain
G.v0[4] = 2.0

# Define synapses with weight w
S = Synapses(G, G, model='w : 1', on_pre='v_post += w')
S.connect(condition='j == i+1 and i < N_pre-1')  # left-to-right connections
S.connect(condition='j == i-1 and i > 0')         # right-to-left connections (bidirectional)
S.w = 1.0

# Run the simulation
run(duration)

# Animate membrane potential propagation
animate_signal_transfer(VM, N)

# Plot spiking activity
figure(figsize=(12, 4))
plot(M.t / ms, M.i, '.k')  # spike time on x-axis, neuron index on y-axis
xlabel('Time (ms)')
ylabel('Neuron index')
title('Spiking Activity of Neurons')
plt.show(block=True)