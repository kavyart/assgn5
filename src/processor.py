import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
from cp_hw2 import lRGB2XYZ
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources


def normalize(I):
    return (I - np.min(I)) / (np.max(I) - np.min(I))


############# INITIALS #############
def load_luminance_stack():
    stack_size = 7
    width = 0
    height = 0

    tuple_I = tuple()
    for k in range(stack_size):
        I = io.imread('assgn5/data/input_'+str(k+1)+'.tif') #[::N, ::N]
        # I = io.imread('assgn5/data/pooh/pooh'+str(k+1)+'.tiff')[750:3500:8, 1500:4500:8]
        # I = io.imread('assgn5/data/bear/bear'+str(k+1)+'.tiff')[1100:3100:8, 1300:4300:8]
        # I = io.imread('assgn5/data/pin/pin'+str(k+1)+'.tiff')[1100:2950:12, 300:-200:12]
        if (k==0):
            save = normalize(I) * 255
            save_ubyte = save.astype(np.ubyte)
            io.imsave('calibrated_image.png', save_ubyte)
        (height, width, _) = I.shape
        I_XYZ = lRGB2XYZ(I)
        I_lum = I_XYZ[:,:,1].flatten()
        tuple_I += (I_lum,)

    I_stack = np.vstack(tuple_I)
    return I_stack, height, width


############# UNCALIBRATED PHOTOMETRIC STEREO #############
Q = np.array([[1, 0.1, 0.3],
              [0, 1, 0.5],
              [0, 0, 1]])

def uncalibrated_ps():
    I, H, W = load_luminance_stack()

    u, s, vh = np.linalg.svd(I, full_matrices=False)

    clippedVH = vh[:3,:]
    clippedS = s[:3]

    B_e = np.sqrt(clippedS)[:, None] * clippedVH

    A_e = np.linalg.norm(B_e, axis=0)

    N_e = B_e / A_e

    return B_e, A_e, N_e

def ps_with_q():
    B_e, _, _ = uncalibrated_ps()
    Q_mat = np.linalg.inv(Q).T

    B_Q = np.matmul(Q_mat, B_e)

    A_Q = np.linalg.norm(B_Q, axis=0)

    N_Q = B_Q / A_Q

    return B_Q, A_Q, N_Q


############# ENFORCING INTEGRABILITY #############
mu = -0.1
nu = 0.1
lamb = 1.5
G = np.array([[ 1,  0,    0],
              [ 0,  1,    0],
              [mu, nu, lamb]])

G_F = np.array([[1, 0,  0],
                [0, 1,  0],
                [0, 0, -1]])

def enforce_integ(B_e, height, width):
    sigma = 12 # 2, 4, 5, 12, 14, 15, 16
    B_e_reshape = B_e.swapaxes(0,1).reshape((height,width,3))
    B_e_blur = np.zeros(B_e_reshape.shape)
    for i in range(3):
        B_e_blur[:, :, i] = gaussian_filter(B_e_reshape[:, :, i], sigma) 

    d_y, d_x, _ = np.gradient(B_e_blur)
    d_y = d_y.reshape((height*width,3)).swapaxes(0,1)
    d_x = d_x.reshape((height*width,3)).swapaxes(0,1)

    B_e1, B_e2, B_e3 = B_e[0], B_e[1], B_e[2]
    d_y1, d_y2, d_y3 = d_y[0], d_y[1], d_y[2]
    d_x1, d_x2, d_x3 = d_x[0], d_x[1], d_x[2]

    A_1 =  np.multiply(B_e1, d_x2) - np.multiply(B_e2, d_x1)
    A_2 =  np.multiply(B_e1, d_x3) - np.multiply(B_e3, d_x1)
    A_3 =  np.multiply(B_e2, d_x3) - np.multiply(B_e3, d_x2)
    A_4 = -np.multiply(B_e1, d_y2) + np.multiply(B_e2, d_y1)
    A_5 = -np.multiply(B_e1, d_y3) + np.multiply(B_e3, d_y1)
    A_6 = -np.multiply(B_e2, d_y3) + np.multiply(B_e3, d_y2)
    A = np.array([A_1, A_2, A_3, A_4, A_5, A_6]).T

    u, s, vh = np.linalg.svd(A, full_matrices=False)

    xs = vh[-1, :]
    delta = np.array([[-xs[2], xs[5], 1],
                      [xs[1], -xs[4], 0],
                      [-xs[0], xs[3], 0]])
    print(delta)
    delta_inv = np.linalg.inv(delta)

    B_tilde = np.matmul(delta_inv, B_e)
    
    B_tilde = np.matmul(np.linalg.inv(G_F.T), B_tilde)
    B_tilde = np.matmul(np.linalg.inv(G).T, B_tilde)

    a = np.linalg.norm(B_tilde, axis=0)

    N = B_tilde / a
    
    return B_tilde, a, N


############# NORMAL INTEGRATION #############
def normal_integration(N, H, W, epsilon):
    N = N.T.reshape((H,W,3))

    d_x = N[:,:,0] / (N[:,:,2] + epsilon)
    d_y = N[:,:,1] / (N[:,:,2] + epsilon)

    # Z = integrate_poisson(d_x, d_y)
    Z = integrate_frankot(d_x, d_y)
    return Z


############# CALIBRATED PHOTOMETRIC STEREO #############
def calibrated_ps(I):
    L = load_sources()  # 7 x 3

    B = np.linalg.lstsq(L, I, rcond=None)[0]    # 3 x P

    a = np.linalg.norm(B, axis=0)

    N = B / a

    return B, a, N


### BONUS ###
############# RESOLVING THE GBR AMBIGUITY #############
### USING PERSPECTIVE CAMERAS ###
def perspective_integration(N, H, W, epsilon):
    # focal length [pixels] = (focal length [mm] / CCD width [mm]) * image width [pixels]
    f = (55 / 97) * 369
    N = N.T.reshape((H,W,3))
    
    p = N[:,:,0] / (N[:,:,2] + epsilon)
    q = N[:,:,1] / (N[:,:,2] + epsilon)

    p_hat = np.zeros((H,W))
    q_hat = np.zeros((H,W))
    for u in range(H):
        for v in range(W):
            denominator = f - (u * p[u,v]) - (v * q[u,v])
            p_hat[u,v] = (f * p[u,v]) / denominator
            q_hat[u,v] = (f * q[u,v]) / denominator

    Z = integrate_frankot(p_hat, q_hat)
    return Z




def transform(I):
    return (I + 1) / 2

def main():

    print("Initializing variables...")
    I, height, width = load_luminance_stack()

    B, a, N = uncalibrated_ps()
    # B, a, N = ps_with_q()
    B, a, N = enforce_integ(B, height, width)
    # B, a, N = calibrated_ps(I)


    # Z = perspective_integration(N, height, width, 10)
    Z = normal_integration(N, height, width, 0.0001)
    fig = plt.figure()
    plt.imshow(Z, cmap='gray')

    fig = plt.figure()
    plt.imshow(a.reshape((height,width)) * 1, cmap='gray')
    fig = plt.figure()
    plt.imshow(transform(N).swapaxes(0,1).reshape((height,width,3)))

    save = normalize(Z) * 255
    save_ubyte = save.astype(np.ubyte)
    io.imsave('calibrated_depth.png', save_ubyte)


    save = transform(N).swapaxes(0,1).reshape((height,width,3)) * 255
    save_ubyte = save.astype(np.ubyte)
    io.imsave('calibrated_normals.png', save_ubyte)
    

    save = np.clip(a.reshape((height,width)) * 0.03, 0, 1) * 255
    save_ubyte = save.astype(np.ubyte)
    io.imsave('calibrated_albedos.png', save_ubyte)
    plt.show()

    
    # Z is an HxW array of surface depths
    H, W = Z.shape
    # x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    x, y = np.meshgrid(np.arange(W-1,-1,-1), np.arange(H-1,-1,-1))
    # set 3D figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.view_init(elev=78, azim=135)
    # ax.view_init(elev=108, azim=51)
    # add a light and shade to the axis for visual effect
    # (use the ‘-’ sign since our Z-axis points down)
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    # display a surface
    # (control surface resolution using rstride and cstride)
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=1, cstride=1)
    # turn off axis
    plt.axis('off')
    # plt.show()
    plt.savefig('surface.png')


    plt.show()


if __name__ == "__main__":
    main()