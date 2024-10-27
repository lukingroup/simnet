import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib as mpl


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

def density_matrix_plot(rho):
    # Plot the absolute values of the density matrix
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels = [r'$|0\rangle$', r'$|1\rangle$']
    qt.matrix_histogram(rho, title="Density Matrix", ax=ax)
    plt.show()

######################## Plotting for Blind experiments ##################################

### density matrix bar plots for universal single qubit gates
### for TXT gate
def plot_from_rho_TXT(rho, title, filename, color, client):

    if client == True:
        target = (241/255, 95/255, 88/255)
        start = (1,1,1)
        diff = np.array([1-241/255, 1-95/255, 1-88/255])
        diff = diff/np.max(diff)

    elif client == False:
        target = (116/255, 48/255, 98/255)
        start = (1,1,1)
        diff = np.array([1-116/255, 1-48/255, 1-98/255])
        diff = diff/np.max(diff)

    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1})
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=15)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([2,2])

    for ii in range(2):
        for jj in range(2):
            hist[ii,jj] = np.abs(rho[ii,jj])
            # print(hist[ii,jj])

    dz = hist.ravel()

    cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    max_height = 1  # get range of colorbars so we can normalize
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    if client == True:
        print("TRue")
        hist = np.array([[0, 0],
                        [0, 1]])

    elif client == False:
        print("hi")
        hist = np.array([[0.5, 0],
                        [0, 0.5]])
    

    zpos_hist = np.abs(np.real(rho[:]))
    dz_hist = hist - np.abs(np.real(rho[:]))
    # print('zpos_hist', zpos_hist)
    # print('dz_hist', dz_hist)


    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos_hist.ravel()

    # Construct arrays with the dimensions for the 4 bars.
    dx = dy = 0.75 * np.ones_like(zpos)
    dz = hist.ravel()

    dz = dz_hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1, edgecolor='grey', color=(0, 0, 1, 0))


    ax.plot([0.05,0.05],[0.05,0.05],[-0.01,1], color='k', linewidth=1)
    #ax.plot([0.0,0.0],[2.3,4.3],[0,0.5], color='k', linewidth=1.5)


    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0,1])
    # ax.set_zticklabels(["0", "", "1"])

    ax.set_xticks([0.5, 1.5])
    ax.set_xlim([0.09,2.2])
    # ax.set_xticklabels(["$-Y$", "$+Y$"])

    ax.set_yticks([0.5, 1.5])
    ax.set_ylim([0.09,2.2])
    # ax.set_yticklabels(["$-Y$", "$+Y$"])

    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.edge = 0

    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
    ax.zaxis._axinfo["grid"]['linewidth'] =  1
    ax.zaxis._axinfo["tick"]['lenght'] =  0

    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(color=(0,0,0,0))

    ax.w_zaxis.linewidth =  1
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    ax.set_title(title)

    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
### the magic one is unused
def plot_from_rho_magic(rho, title, filename, color, client):

    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=15)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([2,2])

    for ii in range(2):
        for jj in range(2):
            hist[ii,jj] = np.abs(rho[ii,jj])

    dz = hist.ravel()

    cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = (116/255, 48/255, 98/255)
    start = (1,1,1)
    diff = np.array([1-116/255, 1-48/255, 1-98/255])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    max_height = 1  # get range of colorbars so we can normalize
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    hist = np.array([[0, 0],
                     [0, 1]])

    zpos_hist = np.abs(np.real(rho[:]))
    dz_hist = hist - np.abs(np.real(rho[:]))

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos_hist.ravel()

    # Construct arrays with the dimensions for the 4 bars.
    dx = dy = 0.75 * np.ones_like(zpos)
    dz = hist.ravel()

    if client == True:
        hist = np.array([[0, 0],
                        [0, 1]])
    elif client == False:
        hist = np.array([[0.5, 0],
                        [0, 0.5]])

    dz = dz_hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1, edgecolor='grey', color=(0, 0, 1, 0))


    ax.plot([0.05,0.05],[0.05,0.05],[-0.01,1], color='k', linewidth=1)
    #ax.plot([0.0,0.0],[2.3,4.3],[0,0.5], color='k', linewidth=1.5)


    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0,1])
    ax.set_zticklabels(["0", "", "1"])

    ax.set_xticks([0.5, 1.5])
    ax.set_xlim([0.09,2.2])
    # ax.set_xticklabels(["-TXT", "+TXT"])
    ax.set_yticks([0.5, 1.5])
    ax.set_ylim([0.09,2.2])
    # ax.set_yticklabels(["-TXT", "+TXT"])

    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.edge = 0

    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
    ax.zaxis._axinfo["grid"]['linewidth'] =  1
    ax.zaxis._axinfo["tick"]['lenght'] =  0

    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(color=(0,0,0,0))

    ax.w_zaxis.linewidth =  1
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
### for Identity gate
def plot_from_rho_Identity(rho, title, filename, color, client):

    if client == True:
        target = (241/255, 95/255, 88/255)
        start = (1,1,1)
        diff = np.array([1-241/255, 1-95/255, 1-88/255])
        diff = diff/np.max(diff)

    elif client == False:
        target = (116/255, 48/255, 98/255)
        start = (1,1,1)
        diff = np.array([1-116/255, 1-48/255, 1-98/255])
        diff = diff/np.max(diff)

    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1})
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=15)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([2,2])

    for ii in range(2):
        for jj in range(2):
            hist[ii,jj] = np.abs(rho[ii,jj])
            # print(hist[ii,jj])

    dz = hist.ravel()

    cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    max_height = 1  # get range of colorbars so we can normalize
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    if client == True:
        print("TRue")
        hist = np.array([[0, 0],
                        [0, 1]])

    elif client == False:
        print("hi")
        hist = np.array([[0.5, 0],
                        [0, 0.5]])
    

    zpos_hist = np.abs(np.real(rho[:]))
    dz_hist = hist - np.abs(np.real(rho[:]))
    # print('zpos_hist', zpos_hist)
    # print('dz_hist', dz_hist)


    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos_hist.ravel()

    # Construct arrays with the dimensions for the 4 bars.
    dx = dy = 0.75 * np.ones_like(zpos)
    dz = hist.ravel()

    dz = dz_hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1, edgecolor='grey', color=(0, 0, 1, 0))


    ax.plot([0.05,0.05],[0.05,0.05],[-0.01,1], color='k', linewidth=1)
    #ax.plot([0.0,0.0],[2.3,4.3],[0,0.5], color='k', linewidth=1.5)


    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0,1])
    # ax.set_zticklabels(["0", "", "1"])

    ax.set_xticks([0.5, 1.5])
    ax.set_xlim([0.09,2.2])
    # ax.set_xticklabels(["$-Y$", "$+Y$"])

    ax.set_yticks([0.5, 1.5])
    ax.set_ylim([0.09,2.2])
    # ax.set_yticklabels(["$-Y$", "$+Y$"])

    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.edge = 0

    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
    ax.zaxis._axinfo["grid"]['linewidth'] =  1
    ax.zaxis._axinfo["tick"]['lenght'] =  0

    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(color=(0,0,0,0))

    ax.w_zaxis.linewidth =  1
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    ax.set_title(title)

    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
### for general gate (unused)
def plot_from_rho(rho, title, filename, color, ON):

    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = 0.5*np.ones([2,2])


        zpos_hist = np.abs(np.real(rho[0:4:3,0:4:3]))
        dz_hist = hist - np.abs(np.real(rho[0:4:3,0:4:3]))

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([2,2])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,0.5], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,0.5], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.25, 0.5])
        ax.set_zlim([0,0.5])
        ax.set_zticklabels(["0", "", "0.5"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,1]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

### density matrix bar plots for two-qubit gates
def plot_from_rho_intranode_client(rho, title, filename, color, ON):

    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = np.array([[0.5,0,0,0.5],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0.5,0,0,0.5]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,1]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

def plot_from_rho_intranode_server(rho, title, filename, color, ON):

    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = 0.5*np.ones([2,2])


        zpos_hist = np.abs(np.real(rho[0:4:3,0:4:3]))
        dz_hist = hist - np.abs(np.real(rho[0:4:3,0:4:3]))

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([2,2])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,0.5], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,0.5], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.25, 0.5])
        ax.set_zlim([0,0.5])
        ax.set_zticklabels(["0", "", "0.5"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

def plot_from_rho_internode_server(rho, title, filename, color, ON):

    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = 0.5*np.ones([2,2])


        zpos_hist = np.abs(np.real(rho[0:4:3,0:4:3]))
        dz_hist = hist - np.abs(np.real(rho[0:4:3,0:4:3]))

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([2,2])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,0.5], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,0.5], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.25, 0.5])
        ax.set_zlim([0,0.5])
        ax.set_zticklabels(["0", "", "0.5"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

def plot_from_rho_internode_client(rho, title, filename, color, ON):

    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = np.array([[0.5,0,0,0.5],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0.5,0,0,0.5]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,1]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)