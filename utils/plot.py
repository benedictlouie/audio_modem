import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.parameters import *

def plot_sent_received_constellation(sent: np.ndarray, received: np.ndarray) -> None:
    """
    Plot the constellation of sent and received symbols.
    Sent symbols are used to color received ones, and ideal sent locations are also shown.
    """

    received = received[:len(sent)]

    # received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))[:,SYMBOLS_PER_BLOCK*0//10:SYMBOLS_PER_BLOCK*1//10].flatten()
    # sent = np.reshape(sent, (-1, SYMBOLS_PER_BLOCK))[:,SYMBOLS_PER_BLOCK*0//10:SYMBOLS_PER_BLOCK*1//10].flatten()
    # received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))[4:8,:].flatten()
    # sent = np.reshape(sent, (-1, SYMBOLS_PER_BLOCK))[4:8,:].flatten()

    unique_symbols = np.unique(sent)
    colors = ['red', 'blue', 'green', 'orange']
    color_map = dict(zip(unique_symbols, colors))

    plt.figure(figsize=(8, 8))

    if MODE < 3:
        # Received symbols marked as dots
        for sym in unique_symbols:
            mask = sent == sym
            plt.scatter(received[mask].real, received[mask].imag,
                        color=color_map[sym], alpha=0.1, label=f'Received')
        # Sent symbols marked with an "x"
        for sym in unique_symbols:
            plt.plot(sym.real, sym.imag, 'x', markersize=12, markeredgewidth=2,
                    color=color_map[sym], label=f'Sent: {sym}')
    else:
        plt.scatter(received.real, received.imag, color='gray', alpha=0.1)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    # Set title
    if MODE < 3:
        accuracy = np.mean(sent == np.sign(received.real) + 1j * np.sign(received.imag))
        plt.title(f'Constellation: Sent vs Received. Accuracy: {accuracy:.4f}.')
    else:
        plt.title(f'Received constellation')

    plt.axis('equal')

    # Animation
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], color='gray', alpha=0.1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    def update(frame):
        sc.set_offsets(np.c_[received[:frame+1].real, received[:frame+1].imag])
        return sc,
    ani = FuncAnimation(fig, update, frames=len(received), interval=1, blit=True)

    plt.show()

def plot_error_per_bin(received: np.ndarray, sent: np.ndarray, filter: np.ndarray) -> None:
    """
    Plot the filter magnitude (in oragne) and the bit error rate (in blue) after applying the filter.
    """

    # Negelct the last DFT blocks
    received = received[: len(sent) // SYMBOLS_PER_BLOCK * SYMBOLS_PER_BLOCK]
    sent = sent[: len(sent) // SYMBOLS_PER_BLOCK * SYMBOLS_PER_BLOCK]

    received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))
    sent = np.reshape(sent, (-1, SYMBOLS_PER_BLOCK))

    received = np.sign(received.real) + 1j * np.sign(received.imag)
    error_rate = np.mean(received != sent, axis=0)
    filter_magnitude = np.abs(np.mean(filter, axis=0))[1: SYMBOLS_PER_BLOCK + 1]
    received_magnitude = np.abs(np.mean(received, axis=0))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.bar(np.arange(SYMBOLS_PER_BLOCK), error_rate, color=color, alpha=0.6, label='Error Rate')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Error Rate', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.plot(np.arange(SYMBOLS_PER_BLOCK), filter_magnitude, color=color, label='Filter Magnitude')
    # ax2.plot(np.arange(SYMBOLS_PER_BLOCK), received_magnitude, 'r-', label='Received Magnitude')
    ax2.set_ylabel('Magnitude', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Error Rate per bin, Filter Magnitude and Received Magnitude')
    fig.tight_layout()
    plt.show()

def plot_received(received: np.ndarray) -> None:
    received = received[: len(received) // SYMBOLS_PER_BLOCK * SYMBOLS_PER_BLOCK]
    received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))
    received = np.mean(np.abs(received), axis=1)
    plt.plot(received)
    plt.xlabel("DFT block")
    plt.ylabel("Mean magnitude")
    plt.show()