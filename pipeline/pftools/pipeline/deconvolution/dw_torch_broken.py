import torch as tr
from typing import Optional, Tuple, List
import numpy as np
import math

BACKING="numpy"
ArrType = tr.Tensor

def fft(arr:tr.Tensor) -> tr.Tensor:
    return tr.fft.rfftn(arr)

def fft_mul_conj(A:tr.Tensor, B:tr.Tensor) -> tr.Tensor:
    """
    C = conj(A)*B
    """
    return tr.conj(A) * B

def fft_convolve_cc_f2(A:tr.Tensor, B:tr.Tensor) -> tr.Tensor:
    return tr.fft.irfftn(A*B)

def fft_convolve_cc(A:tr.Tensor, B:tr.Tensor) -> tr.Tensor:
    M, N, P = A.shape
    C = A*B
    MNP = M*N*P
    return tr.fft.ifft(C)/MNP

def fft_convolve_cc_conj_f2(A:tr.Tensor, B:tr.Tensor) -> tr.Tensor:
    return tr.fft.irfftn(tr.conj(A)*B)

def get_error(y: tr.Tensor, g: tr.Tensor, metric:str='IDIV') -> float:
    if metric == 'MSE':
        error = get_fMSE(y, g)
    elif metric == 'IDIV':
        error = get_fIdiv(y, g)
    return error

def get_fMSE(y:tr.Tensor, g: tr.Tensor) -> float:
    """
    Get mean squared error between input y and guess g, 
    on domain of g
    """
    M, N, P = g.shape
    y_subset = y[:M, :N, :P]
    err = ((y_subset - g)**2).sum()
    return err / (M*N*P) # get average error

def get_fIdiv(y:tr.Tensor, g:tr.Tensor) -> float:
    M, N, P = g.shape
    y_subset = y[:M, :N, :P].ravel()
    g_flat = g.ravel()
    pos_idx = tr.argwhere((y_subset > 0)*(g_flat>0)).ravel()
    I = tr.sum(g_flat[pos_idx]*tr.log(g_flat[pos_idx]/y_subset[pos_idx]) - (g_flat[pos_idx]-y_subset[pos_idx]))
    return I/(M*N*P)

def psf_autocrop(psf:tr.Tensor, im:tr.Tensor, border_quality:int=2, xycropfactor:float=0.001) -> tr.Tensor:
    M,N,P = im.shape
    pM, pN, pP = psf.shape
    psf = psf_autocrop_by_image(psf, im)
    # crop the PSF by removing outer planes that have litter information
    # but only if PSF is larger than the image in some dimension
    if border_quality > 0:
        psf = psf_autocrop_xy(psf, xycropfactor=xycropfactor)
    return psf

def psf_autocrop_by_image(psf:tr.Tensor, im:tr.Tensor, border_quality:int=2) -> tr.Tensor:
    m, n, p = psf.shape
    M, N, P = im.shape
    if border_quality == 0:
        mopt = M
        nopt = N
        popt = P
    else:
        mopt = (M-1)*2 + 1
        nopt = (N-1)*2 + 1
        popt = (P-1)*2 + 1
    if p < popt:
        print("PSF is smaller than image, no cropping")
        return psf
    if (p%2) == 0:
        # raise error that PSF should have odd number of slices
        return None
    if (m > mopt) or (n > nopt) or (p > popt):
        # initialize with everything set to bound
        m0 = n0 = p0 = 0
        m1, n1, p1 = m-1, n-1, p-1
        if m > mopt:
            m0 = (m-mopt)//2
            m1 -= (m-mopt)//2
        if n > nopt:
            n0 = (n-nopt)//2
            n1 -= (n-nopt)//2
        if p > popt:
            p0 = (p-popt)//2
            p1 -= (p-popt)//2
        psf_cropped = psf[m0:m1+1, n0:n1+1, p0:p1+1]
        print(f"PSF Z-crop: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped

def psf_autocrop_xy(psf:tr.Tensor, xycropfactor:float=0.001) -> tr.Tensor:
    m, n, p = psf.shape
    # find the y-z plane with the largest sum
    #psf_arr = np.array(psf)
    sum_over_plane = tr.sum(psf, axis=(1,2))
    maxsum = tr.max(sum_over_plane)
    #maxsum = 0
    #for xx in range(m):
    #    sum_val = 0
    #    for yy in range(n):
    #        for zz in range(p):
    #            sum_val += psf_arr[xx, yy, zz]
    #    maxsum = max(sum_val, maxsum)
    first = -1
    sum_val = 0

    while sum_val < xycropfactor * maxsum:
        first += 1
        sum_val = 0
        for yy in range(n):
            for zz in range(p):
                sum_val += psf[first, yy, zz]
    if first < 1:
        print(f"No XY crop, shape is {list(psf.shape)}")
        return psf
    else:
        psf_cropped = psf[first:m-first, first:n-first, :]
        print(f"PSF XY crop: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped

def psf_autocrop_center_z(psf:tr.Tensor) -> tr.Tensor:
    m, n, p = psf.shape
    midm = (m-1)//2
    midn = (n-1)//2
    midp = (p-1)//2
    maxm, maxn, maxp = np.array(max_idx(psf)).astype(int)
    if midp == maxp:
        print(f"No cropping of PSF in Z, size is {list(psf.shape)}")
        return psf
    else:
        m0 = n0 = 0
        m1, n1 = m-1, n-1
        # identify the max number of planes in either direction
        p0 = maxp
        p1 = maxp
        # start at the max plane
        # this will select as many planes as possible in both directions
        # while keeping the stack symmetric in z
        while (p0 > 1) and (p1+2 < p):
        #while (p0 >= 0) and (p1 < p):
            p0 -= 1
            p1 += 1
        print(f"Brightest at plane {maxp}")
        print(f'Selecting z planes at {p0}:{p1}')
        psf_cropped = psf[m0:m1+1, n0:n1+1, p0:p1+1]
        print(f"Cropping PSF in Z: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped



def initial_guess(M:int, N:int, P:int, wM:int, wN:int, wP:int) -> tr.Tensor:
    """
    Create initial guess: the fft of an image that is 1 in MNP and 0 outside
     * M, N, P is the dimension of the microscopic image
    """
    one = tr.zeros((wM, wN, wP))
    one[:M,:N,:P] = 1
    return fft(one)


def gaussian_kernel_1d(sigma:float) -> tr.Tensor:
    n = 1 # guarantee at least 1
    while math.erf((n+1)/sigma) < (1.0-1e-8):
        n += 1
    N = 2*n + 1
    K = tr.zeros(N)
    mid = int((N-1)/2)
    s2 = sigma**2
    for kk in range(N):
        x = kk-mid
        K[kk] = math.exp(-0.5*(x**2)/s2)
    return K/K.sum()

def gaussian_kernel_3d(sigma_x:float, sigma_y:float, sigma_z:float) -> tr.Tensor:
    n_x = 1 # guarantee at least 1
    while math.erf((n_x+1)/sigma_x) < (1.0-1e-8):
        n_x += 1
    n_y = 1
    while math.erf((n_y+1)/sigma_y) < (1.0-1e-8):
        n_y += 1
    n_z = 1
    while math.erf((n_z+1)/sigma_z) < (1.0-1e-8):
        n_z += 1
 
    N_x = 2*n_x + 1
    N_y = 2*n_y + 1
    N_z = 2*n_z + 1
    mid_x = float((N_x-1)/2)
    mid_y = float((N_y-1)/2)
    mid_z = float((N_z-1)/2)
    x = tr.arange(0, n_x).float() - mid_x
    y = tr.arange(0, n_y).float() - mid_y
    z = tr.arange(0, n_z).float() - mid_z
    x,y,z = tr.meshgrid([x,y,z])
    gauss_x = tr.exp(-0.5 * (x/sigma_x)**2)
    gauss_y = tr.exp(-0.5 * (y/sigma_y)**2)
    gauss_z = tr.exp(-0.5 * (z/sigma_z)**2)
    K = gauss_x * gauss_y * gauss_z
    return K/tr.sum(K)

def max_idx(arr:tr.Tensor) -> tr.Tensor:
    return tr.argwhere(arr == arr.max()).ravel()

def circshift(arr:tr.Tensor,shifts:tr.Tensor) -> tr.Tensor:
    # shift each axis in turn
    #for i in range(len(shifts)):
    #    arr = tr.roll(arr, int(shifts[i]), axis=i)
    shifts = tuple(shifts)
    return tr.roll(arr, shifts, (0,1,2))

def insert(T:tr.Tensor, F:tr.Tensor) -> tr.Tensor:
    """
    Insert [f1 x f2 x f3] into T [t1 x t2 x t3] in the 'upper left' corner
    """
    if T.ndim == 2:
        F1, F2 = F.shape
        T[:F1, :F2] = F
    elif T.ndim == 3:
        F1, F2, F3 = F.shape
        T[:F1, :F2, :F3] = F 
    return T

def gsmooth_aniso(arr:tr.Tensor, lsigma:float, asigma:float, padding='constant') -> tr.Tensor:
    M, N, P = arr.shape
    K = gaussian_kernel_3d(lsigma, lsigma, asigma)

    kx, ky, kz = K.shape
    arr_padded = tr.nn.functional.pad(arr, (kx,kx,ky,ky,kz,kz), mode=padding)
    arr_f = tr.fft.fftn(arr_padded)
 
    temp = tr.zeros_like(arr_padded) 
    temp = insert(temp, K)
    maxi = tuple([-int(i) for i in max_idx(temp)])
    #print(maxi)
    
    temp = circshift(temp, maxi)
    k_f = tr.fft.fftn(temp)
    result = tr.real(tr.fft.ifftn(arr_f * k_f))
    return result[kx:(M+kx), ky:(N+ky), kz:(P+kz)]

def gsmooth(arr:tr.Tensor, gsigma:float) -> tr.Tensor:
    """
    Gaussian smooth array
    """
    return gsmooth_aniso(arr, gsigma, gsigma)

def get_midpoint(arr:tr.Tensor) -> Tuple[int, int, int]:
    m,n,p = arr.shape
    return int((m-1)/2), int((n-1)/2), int((p-1)/2)

def prefilter(im:tr.Tensor, psf:tr.Tensor, psigma:float=0) -> Tuple[tr.Tensor, tr.Tensor]:
    if psigma <= 0:
        return im, psf
    else:
        return gsmooth(im, psigma), gsmooth(psf, psigma)

def compute_tile_positions(im:tr.Tensor, max_size:int, overlap:int) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                                                            List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Return a list of the x,y indices of the tiles
    """
    M,N,P = im.shape
    tile_pos_with_overlap = []
    tile_pos_without_overlap = []
    n_tiles_x = np.ceil(float(M)/float(max_size)).astype(int)
    n_tiles_y = np.ceil(float(N)/float(max_size)).astype(int)
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            tile_start_x = max(0, i*max_size-overlap)
            tile_stop_x = min((i+1)*max_size+overlap, M)
            tile_start_y = max(0, j*max_size-overlap)
            tile_stop_y = min((j+1)*max_size+overlap, N)
            tile_pos_with_overlap.append(((tile_start_x, tile_stop_x), (tile_start_y, tile_stop_y)))

            tile_start_x = max(0, i*max_size)
            tile_stop_x = min((i+1)*max_size, M)
            tile_start_y = max(0, j*max_size)
            tile_stop_y = min((j+1)*max_size, N)
            tile_pos_without_overlap.append(((tile_start_x, tile_stop_x), (tile_start_y, tile_stop_y)))
    return tile_pos_with_overlap, tile_pos_without_overlap

def run_dw_tiled(im:tr.Tensor, psf:tr.Tensor, 
    tile_max_size:int=256, tile_padding:int=40,
    n_iter:int=10, alphamax:float=10, bg:Optional[float]=None, 
    relax:int=0, psigma:int=0, border_quality:int=2,
    positivity:bool=True,method:str='shb' ) -> tr.Tensor:
    M, N, P = im.shape

    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im *= 1000/im.max()

    # normalize PSF
    psf /= psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf[mid_x, mid_y, mid_z] += relax
        psf /= psf.sum()

    # split image into N tiles
    pos_with_overlap, pos_without_overlap = compute_tile_positions(im, tile_max_size, tile_padding) 
    decon_img = tr.zeros_like(im)
    for i in range(len(pos_with_overlap)):
        # get tile with padding
        (min_x_overlap, max_x_overlap), (min_y_overlap, max_y_overlap) = pos_with_overlap[i]
        #print(min_x_overlap, max_x_overlap, min_y_overlap, max_y_overlap)
        curr_tile = im[min_x_overlap:max_x_overlap, min_y_overlap:max_y_overlap, :]
        # run decon on tile with padding
        #print(curr_tile.shape, psf.shape)
        res = decon(curr_tile, psf, psigma=psigma, n_iter=n_iter, alphamax=alphamax, bg=bg, border_quality=border_quality, 
                    positivity=positivity, method=method)
        # get rid of padding
        (min_x, max_x), (min_y, max_y) = pos_without_overlap[i]
        if min_x == 0:
            crop_x_min = 0
        else:
            crop_x_min = tile_padding
        if max_x == M:
            crop_x_max = res.shape[0]
        else:
            crop_x_max = res.shape[0]-tile_padding
        if min_y == 0:
            crop_y_min = 0
        else:
            crop_y_min = tile_padding
        if max_y == N:
            crop_y_max = res.shape[1]
        else:
            crop_y_max = res.shape[1]-tile_padding
        res_cropped = res[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]
        #print(res.shape, res_cropped.shape, tile_padding, min_x, max_x, min_y, max_y)
        decon_img[min_x:max_x, min_y:max_y, :] = res_cropped
    return decon_img

def run_dw(im:tr.Tensor, psf:tr.Tensor, 
    n_iter:int=10, alphamax:float=10, bg:Optional[float]=None, 
    relax:int=0, psigma:int=0, border_quality:int=2,
    positivity:bool=True,method:str='shb') -> tr.Tensor:
    M, N, P = im.shape

    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im *= 1000/im.max()
        
    # normalize PSF
    psf /= psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf[mid_x, mid_y, mid_z] += relax
        psf /= psf.sum()
    
    psf /= psf.sum()
    im, psf = prefilter(im, psf, psigma) 
    return decon(im, psf, psigma, n_iter, alphamax, bg, border_quality, positivity, method)


def decon(im:tr.Tensor, psf:tr.Tensor, psigma:int=3, n_iter:int=10, alphamax:float=10, 
          bg:Optional[float]=None, border_quality:int=2, positivity:bool=True, method:str='rl',err_thresh:float=0.01) -> tr.Tensor:
    # auto compute background
    if bg is None:
        bg = im.min()
        if bg < 1e-2:
            bg = 1e-2

    M, N, P = im.shape
    pM, pN, pP = psf.shape 
    wM = M + pM - 1
    wN = N + pN - 1
    wP = P + pP - 1

    if border_quality == 1:
        wM = M + (pM + 1)/2
        wN = N + (pN + 1)/2
        wP = P + (pP + 1)/2

    elif border_quality == 0:
        wM = max(M, pM)
        wN = max(N, pN)
        wP = max(P, pP)

    Z = tr.zeros((wM, wN, wP))

    # insert the PSF into the larger image
    Z = insert(Z, psf)

    # shift PSF so midpoint is at (0,0,0)
    Z = circshift(Z, -max_idx(Z))

    # PSF FFT
    cK = fft(Z)
    del Z
    sigma = 0.01
    if border_quality > 0:
        F_one = initial_guess(M, N, P, wM, wN, wP)
    
        W = tr.fft.irfftn(fft_mul_conj(cK, F_one))
        idx = W>sigma
        W[idx] = 1/W[idx]
        W[~idx] = 0
    
    sumg = im.sum() 
    # x is initial guess, initially previous iteration xp is set to be the same
    x = tr.ones((wM, wN, wP)) * sumg/(wM*wN*wP)
    xp = x
    prev_err = 1e6
    for i in range(n_iter):
        if method == 'shb':
            # Eq. 10 in SHB paper
            alpha = (i-1.0)/(i+2.0)
            if alpha < 0:
                alpha = 0
            if alpha > alphamax:
                alpha = alphamax
            # current guess, based on update from previous
            p = xp
            # p^k in Eq. 7 o SHB paper
            # estimate gradient based on scaled difference from previous round
            p = x + alpha*(x - xp) 
            # set pixels less than background to background
            p[p < bg] = bg
            
            # optionally smooth image
            if psigma > 0:
                x = gsmooth(x, psigma)

            xp_temp, err = iter_shb(im, cK, p, W)
            # swap to update
            xp = x
            x = xp_temp
        
        elif method == 'rl':
            x, err = iter_rl(im, cK, xp, bg, W)
            xp = x
        
        print(f"Iter: {i}, Err: {float(err):2.2f}, Delta: {float(prev_err-err):2.2f}")
        if prev_err-err < err_thresh:
            break

        prev_err = err
        if positivity and bg > 0:
            # this isn't in RL function
            x[x < bg] = bg

    x = xp
    # crop to corner subregion
    return x[:M,:N,:P]

def iter_rl(im: tr.Tensor, fftPSF:tr.Tensor, f:tr.Tensor, bg:float, W:Optional[tr.Tensor]=None) -> tr.Tensor:
    M, N, P = im.shape
    wM, wN, wP = f.shape
    F = fft(f)
    y = tr.fft.irfftn(fftPSF * F)#fft_convolve_cc_f2(fftPSF, F)
    error = get_error(y, im)
    # crop down to size of image
    y_subset = y[:M,:N,:P]
    idx = y_subset > 0
    y_subset[idx] = im[idx]/y_subset[idx] 
    y_subset[~idx] = bg
    # set everything outside of image to 1e-6
    #y[M:, N:, P:] = 1e-6
    y = tr.ones_like(y)*1e-6
    # set everything within image
    y[:M, :N, :P] = y_subset
    # convolve with PSF for next iteration
    F_sn = fft(y)
    x = tr.fft.irfftn(tr.conj(fftPSF) * F_sn)#fft_convolve_cc_conj_f2(fftPSF, F_sn)
    if W is None:
        x *= f
    else:
        x *= f * W
    return x, error

def iter_shb(im:tr.Tensor, cK:tr.Tensor, pK:tr.Tensor, W:Optional[tr.Tensor]=None) -> Tuple[tr.Tensor, float]:
    """Iteration of SHB

    Args:
        im (tr.Tensor): Input image
        cK (tr.Tensor): fft(psf) 
        pK (tr.Tensor): p_k, estimation of gradient
        W (tr.Tensor): Bertero weights

    Returns:
        tr.Tensor: _description_
    """
    M, N, P = im.shape
    pK_F = fft(pK)
    # convolve with PSF
    y = fft_convolve_cc_conj_f2(cK, pK_F)
    error = get_error(y, im)
    mindiv = 1e-6 # smallest allowed divisor
    y_subset = y[:M, :N, :P]
    y_subset[tr.abs(y_subset) < mindiv] = tr.sign(y_subset[tr.abs(y_subset) < mindiv])*mindiv
    y_subset = im/y_subset
    y = tr.zeros_like(y)
    y[:M, :N, :P] = y_subset
    #y[M:, N:, P:] = 0
    Y = fft(y)
    # convolve with PSF 
    x = fft_convolve_cc_conj_f2(cK, Y)
    if W is not None:
        x *= pK * W
    else:
        x *= pK
    return x, error
