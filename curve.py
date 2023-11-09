import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def graph1():
  # Define parameters for the Gaussian distribution
  mean = 1.25  # Mean for the Gaussian
  std_dev = 0.3  # Standard deviation for the Gaussian

  # Define the x-axis values (planetary radii in Earth radii)
  x = np.linspace(0, 3, 1000)

  # Calculate the Gaussian distribution
  gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

  # Create a mask for the flat top region
  flat_top_mask = (x >= 0.5) & (x <= 2)

  # Apply the mask to create the flat top
  gaussian[flat_top_mask] = 1  # Set the values inside the mask to 1

  # Plot the distribution
  plt.plot(x, gaussian)
  plt.xlabel("Planetary Radius (Earth Radii)")
  plt.ylabel("Habitability Score")
  plt.title("Graph 1")
  plt.ylim(0, 1.1)
  plt.show()

def graph2():
  mean_peak = 1.0  # Mean for the peak
  std_dev_peak = 0.2  # Standard deviation for the peak

  mean_flat_top = 1.25  # Mean for the flat top
  std_dev_flat_top = 0.3  # Standard deviation for the flat top

  # Define the x-axis values (planetary radii in Earth radii)
  x = np.linspace(0, 3, 1000)

  # Calculate the Gaussian distributions for the peak and flat top
  peak = (1 / (std_dev_peak * np.sqrt(2 * np.pi))) * np.exp(-((x - mean_peak) ** 2) / (2 * std_dev_peak ** 2))
  flat_top = (1 / (std_dev_flat_top * np.sqrt(2 * np.pi))) * np.exp(-((x - mean_flat_top) ** 2) / (2 * std_dev_flat_top ** 2))

  # Combine the peak and flat top distributions
  y = peak + flat_top

  # Ensure that the combined distribution is normalized to have an area under the curve equal to 1
  y /= np.trapz(y, x)


  # Plot the distribution
  plt.plot(x, y)
  plt.xlabel("Planetary Radius (Earth Radii)")
  plt.ylabel("Habitability Score")
  plt.title("Graph 2")
  plt.ylim(0, 1)
  plt.show()



def graph3():
  # Define parameters for the Gaussian distribution
  mean = 1.25  # Mean for the Gaussian
  std_dev = 0.3  # Standard deviation for the Gaussian

  # Define the x-axis values (planetary radii in Earth radii)
  x = np.linspace(0, 3, 1000)

  # Calculate the Gaussian distribution
  gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

  # Create a mask for the flat top region
  flat_top_mask = (x >= 0.5) & (x <= 2)

  # Apply the mask to create the flat top
  gaussian[flat_top_mask] = 1  # Set the values inside the mask to 1

  # Smooth the edges using a sigmoid transition
  sigmoid_width = 0.1  # Adjust the width of the sigmoid transition
  transition = 1 / (1 + np.exp(-((x - 0.5) / sigmoid_width)))
  gaussian = gaussian * transition

  # Plot the distribution
  plt.plot(x, gaussian)
  plt.xlabel("Planetary Radius (Earth Radii)")
  plt.ylabel("Habitability Score")
  plt.title("Graph 3")
  plt.ylim(0, 1)
  plt.show()

def graph4():

  # Parameters
  f_0 = 1
  w = 1.5
  n = 6

  # Create data points
  x = np.linspace(-1.5, 5, 1000)


  # Define the custom function using erf
  def custom_function(x, f_0, w, n):
      return 0.5 * f_0 * (erf(n * (x + 0.5 * w)) - erf(n * (x - 0.5 * w)))

  # Plot the custom functions for different values of n
  plt.figure(figsize=(10, 6))
  y = custom_function(x, f_0, w, n)
  x = x + 1.5
  plt.plot(x, y, label=f'n = {n}')

  plt.xlabel("Radius (In Earth Radii)")
  plt.ylabel("Habitability (Probability between 0 to 1)")
  plt.title("Custom Gaussian with Flat Top")
  plt.legend()
  plt.grid(True)
  plt.show()

#graph1()
#graph2()
#graph3()
#graph4()



def graph_radius(left_n, right_n):

    # Parameters
    f_0 = 1
    w = 1.5
    n = 6

    # Create data points
    x = np.linspace(-1.5, 5, 1000)

    # Define the custom function using erf
    def custom_function(x, f_0, w, n, left_n, right_n):
        left_part = 0.5 * f_0 * (erf(left_n * (x + 0.5 * w)) - erf(left_n * (x - 0.5 * w)))
        right_part = 0.5 * f_0 * (erf(right_n * (x + 0.5 * w)) - erf(right_n * (x - 0.5 * w)))
        return left_part * (x <= 0) + right_part * (x > 0)

    # Plot the custom function with the specified left_n and right_n
    plt.figure(figsize=(10, 6))
    y = custom_function(x, f_0, w, n, left_n, right_n)
    x = x + 1.5
    plt.plot(x, y, label=f'left curvature const. = {left_n}, right curvature const. = {right_n}')

    plt.xlabel("Radius (In Earth Radii)")
    plt.ylabel("Habitability (Probability between 0 to 1)")
    plt.title("Custom Gaussian with Flat Top")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
left_n = 3.5  # Adjust the left curvature
right_n = 3.5  # Adjust the right curvature
#graph_radius(left_n, right_n)


def graph_eccentricity(left_n, right_n):

    # Parameters
    f_0 = 1
    w = 80
    n = 2

    # Create data points
    x = np.linspace(0, 100, 1000)
    #plt.gca().invert_xaxis()
    #newx = 100 - x

    

    # Define the custom function using erf
    def custom_function(x, f_0, w, n, left_n, right_n):
        left_part = 0.5 * f_0 * (erf(left_n * (x + 0.5 * w)) - erf(left_n * (x - 0.5 * w)))
        right_part = 0.5 * f_0 * (erf(right_n * (x + 0.5 * w)) - erf(right_n * (x - 0.5 * w)))
        return left_part * (x <= 0) + right_part * (x > 0)

    # Plot the custom function with the specified left_n and right_n
    plt.figure(figsize=(10, 6))
    y = custom_function(x, f_0, w, n, left_n, right_n)
    plt.plot(x, y, label=f'left curvature const. = {left_n}, right curvature const. = {right_n}')

    # Set the x-axis ticks and labels
    x_ticks = np.arange(0, 101, 20)
    x_tick_labels = [str(100 - tick) for tick in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)

    plt.xlabel("Eccentricity due to Orbit(time spent in HZ)")
    plt.ylabel("Habitability (Probability between 0 to 1)")
    plt.title("Custom Gaussian with Flat Top")
    plt.legend()
    plt.grid(True)

    plt.show()

# Example usage:
left_n = 8  # Adjust the left curvature
right_n = 0.2  # Adjust the right curvature
graph_eccentricity(left_n, right_n)





def graph_starlife(left_n, right_n):

    # Parameters
    f_0 = 1
    w = 7
    n = 8

    # Create data points
    x = np.linspace(0, 5, 1000)
    newx = 5 - x

    # Define the custom function using erf
    def custom_function(x, f_0, w, n, left_n, right_n):
        left_part = 0.5 * f_0 * (erf(left_n * (x + 0.5 * w)) - erf(left_n * (x - 0.5 * w)))
        right_part = 0.5 * f_0 * (erf(right_n * (x + 0.5 * w)) - erf(right_n * (x - 0.5 * w)))
        return left_part * (x <= 0) + right_part * (x > 0)

    # Plot the custom function with the specified left_n and right_n
    plt.figure(figsize=(10, 6))
    y = custom_function(x, f_0, w, n, left_n, right_n)
    plt.plot(newx, y, label=f'left curvature const. = {left_n}, right curvature const. = {right_n}')

    plt.xlabel("Time in HZ due to M-star lifetime and evolution (in Gya)")
    plt.ylabel("Habitability (Probability between 0 to 1)")
    plt.title("Custom Gaussian with Flat Top")
    plt.legend()
    plt.grid(True)

    plt.show()

# Example usage:
left_n = 4.5# Adjust the left curvature
right_n = 4.5  # Adjust the right curvature
graph_starlife(left_n, right_n)