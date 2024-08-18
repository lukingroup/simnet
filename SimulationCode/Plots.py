import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from mpl_toolkits.mplot3d import Axes3D 


""" Bell state tomography """
def bell_state_barplotZZ(data):

    fig, ax = plt.subplots(figsize=[14, 8], nrows=1, ncols=1)
    ax.set_title("ZZ")
    ax.bar(np.arange(4), data, color = 'lightpink', edgecolor='lightpink', alpha=.5)
    ax.bar(np.arange(4), [0.5, 0, 0, 0.5], fill=False, color = 'hotpink', edgecolor='hotpink', alpha=.5)
    fig.subplots_adjust(hspace=0.5)

    plt.show()
    return

def bell_state_barplotXX(data):

    fig, ax = plt.subplots(figsize=[14, 8], nrows=1, ncols=1)
    ax.set_title("XX")
    ax.bar(np.arange(4), data, color = 'lightpink', edgecolor='lightpink', alpha=.5)
    ax.bar(np.arange(4), [0, 0.5, 0.5, 0], fill=False, color = 'hotpink', edgecolor='hotpink', alpha=.5)
    fig.subplots_adjust(hspace=0.5)

    plt.show()
    return

def bell_state_barplotYY(data):

    fig, ax = plt.subplots(figsize=[14, 8], nrows=1, ncols=1)
    ax.set_title("YY")
    ax.bar(np.arange(4), data, color = 'lightpink', edgecolor='lightpink', alpha=.5)
    ax.bar(np.arange(4), [0.5, 0, 0, 0.5], fill=False, color = 'hotpink', edgecolor='hotpink', alpha=.5)
    fig.subplots_adjust(hspace=0.5)

    plt.show()
    return


""" 2D plots """
def plot2d(data, x_axis, y_axis):
    plt.plot(data[0], data[1], label = '')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.yscale('linear')
    plt.xscale('linear')    

""" Table """
def table_single_node(ZZ, XX, YY):
    # Sample data for a 3x5 table
    # Sample data for a 3x5 table
    data = {
        '+ X init': [40, 50, 60],
        '- X init': [0, 0, 90],
        '+ Y init': [100, 110, 120],
        '- Y init': [130, 140, 150],
        '+ Z init': [100, 110, 120],
        '- Z init': [130, 140, 150]
    }

    # Creating a DataFrame
    df = pd.DataFrame.from_dict(data, orient = 'index')
    df.columns = ['X meas', 'Y meas', 'Z meas']

    # Display the DataFrame
    print("Table displayed in Python environment:")
    print(df)

    # File path for the output
    file_path = '/Users/azizasuleymanzade/Dropbox (Personal)/AzizaOnly/LukinLab/BlindComputing/SimulationCode_Aziza/SimulationCode/OutputFiles/BlindComputing/TableSingleGates.txt'

    # Write the DataFrame to a text file
    with open(file_path, 'w') as file:
        file.write(df.to_string(index=True))

    return 1

def density_matrix_plot (rho):
    # Plot the absolute values of the density matrix
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels = [r'$|0\rangle$', r'$|1\rangle$']
    qt.matrix_histogram(rho, title="Density Matrix", ax=ax)
    plt.show()