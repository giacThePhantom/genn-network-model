import numpy as np
import matplotlib
import scipy as sp


def compute_sdf_for_population(spike_matrix, sigma, dt):
    kernel = np.arange(-3*sigma, +3*sigma, dt)
    kernel = np.exp(-np.power(kernel, 2)/(2*sigma*sigma))
    kernel = kernel/(sigma*np.sqrt(2.0*np.pi))*1000.0
    res = np.apply_along_axis(lambda m : sp.signal.convolve(m, kernel, mode='same'), axis = 1, arr=spike_matrix)
    return res


def sdf_glomerulus_avg(sdf_matrix, n_glomeruli):
    """
    Get the average activation intensity across n-sized groups of glomeruli
    for each timestep.

    Arguments
    ---------
    sdf_matrix : np.ndarray
        an SDF matrix obtained by eg. `make_sdf`
    glomerulus_dim : int
        How many neurons per glomerulus
    """
    res = np.zeros((n_glomeruli, sdf_matrix.shape[1]))
    glomerulus_dim = sdf_matrix.shape[0] // n_glomeruli
    for i in range(n_glomeruli):
        res[i, :]= np.mean(sdf_matrix[glomerulus_dim*i:glomerulus_dim*(i+1), :],axis=0)
    return res




def plot_sdf_heatmap(sdf_average, t_start, t_end, dt, pop, subplot):
    res = subplot.imshow(sdf_average, vmin = 0, vmax = 100, cmap = 'plasma')
    subplot.set_aspect((t_end-t_start)//10)
    subplot.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([3000*i//dt for i in range(int(t_end-t_start)//3000 + 1)]))
    subplot.set_xticklabels([f"{t_start * (i + 1):.1g}" for i in range(int(t_end-t_start)//3000 + 1)], rotation=45, fontsize = 8)
    subplot.set_title(pop)
    subplot.set_xlabel("Time [ms]")
    subplot.set_ylabel("Glomeruli")
    return res

def plot_sdf_over_time_outliers(sdf_matrix_avg, subplot):
    glomeruli_mean_sdf = np.mean(sdf_matrix_avg, axis = 1)
    global_mean = np.mean(glomeruli_mean_sdf)
    global_sdt = np.std(glomeruli_mean_sdf)
    glomeruli_of_interest = []

    for (i, mean_sdf) in enumerate(glomeruli_mean_sdf):
        if np.abs(mean_sdf - global_mean) > global_sdt:
            glomeruli_of_interest.append(i)

    for i in glomeruli_of_interest:
        subplot.plot(sdf_matrix_avg[i, :], label = f"Glomerulus {i}")
    subplot.legend()
