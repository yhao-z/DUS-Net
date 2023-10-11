import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import scipy.io as scio


def mask3d(nx, ny, nz, center_r=[15, 15, 0], undersampling=0.5):
    # create undersampling mask
    mask_shape = np.array([nx, ny, nz])
    Npts = mask_shape.prod()  # total number of data points
    k = int(round(Npts * undersampling))  # undersampling
    ri = np.random.choice(Npts, k, replace=False)  # index for undersampling
    ma = np.zeros(Npts)  # initialize an all zero vector
    ma[ri] = 1  # set sampled data points to 1
    mask = ma.reshape(mask_shape)

    flag_centerfull = 1
    # x center, k-space index range
    if center_r[0] > 0:
        cxr = np.arange(-center_r[0], center_r[0] + 1) + mask_shape[0] // 2
    elif center_r[0] == 0:
        cxr = np.arange(mask_shape[0])
    else:
        flag_centerfull = 0
    # y center, k-space index range
    if center_r[1] > 0:
        cyr = np.arange(-center_r[1], center_r[1] + 1) + mask_shape[1] // 2
    elif center_r[1] == 0:
        cyr = np.arange(mask_shape[1])
    else:
        flag_centerfull = 0
    # z center, k-space index range
    if center_r[2] > 0:
        czr = np.arange(-center_r[2], center_r[2] + 1) + mask_shape[2] // 2
    elif center_r[2] == 0:
        czr = np.arange(mask_shape[2])
    else:
        flag_centerfull = 0

    # full sampling in the center kspace
    if flag_centerfull != 0:
        mask[np.ix_(cxr, cyr, czr)] = \
            np.ones((cxr.shape[0], cyr.shape[0], czr.shape[0]))  # center k-space is fully sampled
    return mask


def generate_mask(res, uds_ratio_lines, type):
    n1, n2, n3 = res

    # variable density random 2d sampling
    if type == 'vds':
        mask = genrate_binary_sampling_map(n1, n2, uds_ratio_lines, n3)
    elif type == 'vds_y':
        mask = genrate_ylines_sampling_map(n1, n2, uds_ratio_lines, n3)
    elif type == 'uds':
        omega = np.random.rand(n1, n2, n3) < uds_ratio_lines
        mask = np.zeros((n1, n2, n3))
        mask[omega] = 1
    elif type == 'uds_y':
        raws = np.rint(n1 * uds_ratio_lines)
        ind_sample = np.random.randint(n1, size=(int(raws), n3))
        mask = np.zeros((n1, n2, n3))
        for i in range(n3):
            mask[ind_sample[:, i], :, i] = 1
    elif type == 'radial':
        mask = strucrand(n1, n2, n3, uds_ratio_lines)
    elif type == 'vista':
        mask = vista_mask(uds_ratio_lines)
    else:
        raise Exception('type needs to be\n '
                        'vds   ----> variable density random 2d sampling\n'
                        'vds_x ----> variable density randome x sampling\n'
                        'uds   ----> uniform density random 2d sampling\n'
                        'uds_x ----> uniform density randome x sampling\n'
                        'radial----> radial sampling')
    return mask


def genPDF(imSize, p, pctg, distType=2, radius=0, disp=0):
    minval = 0
    maxval = 1
    val = 0.5
    if isinstance(imSize,int):
        imSize = [imSize]
    if len(imSize) == 1:
        imSize.append(1)

    imSize = np.array(imSize)
    sx = imSize[0]
    sy = imSize[1]
    PCTG = np.floor(pctg * sx * sy)

    if np.sum(imSize == 1) == 0:
        x, y = np.meshgrid(np.linspace(-1, 1, sy), np.linspace(-1, 1, sx))
        if distType == 1:
            r = np.max(np.abs(x), np.abs(y))
        else:
            r = np.sqrt(x ** 2 + y ** 2)
            r = r / np.max(np.abs(r[:]))
    else:
        r = np.abs(np.linspace(-1, 1, np.maximum(sx, sy)))

    idx = r < radius
    pdf = (1 - r) ** p
    pdf[idx] = 1
    if np.floor(np.sum(pdf[:])) > PCTG:
        raise Exception('infeasible without undersampling dc, increase p')

    while 1:
        val = minval / 2 + maxval / 2
        pdf = (1 - r) ** p + val
        pdf[pdf > 1] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf[:]))
        if N > PCTG:
            maxval = val
        elif N < PCTG:
            minval = val
        else:
            break

    if disp:
        plt.figure()
        plt.subplot(211)
        plt.imshow(pdf)
        if np.sum(imSize == 1) == 0:
            plt.subplot(212)
            plt.plot(pdf[int(np.size(pdf, axis=0) / 2) + 1, :])
        else:
            plt.subplot(212)
            plt.plot(pdf)

    return pdf, val


def genrate_binary_sampling_map(n1, n2, undersampling_ratio, n3):
    sampling_mat = np.zeros((n1, n2, n3), dtype=int)
    pdf_vardens_cut, _ = genPDF([n1, n2], 9, undersampling_ratio)
    for i in range(n3):
        r_mat = np.random.rand(n1, n2)
        pdf_vardens2 = r_mat * pdf_vardens_cut
        pdf_vardens3 = pdf_vardens2.ravel()
        b = np.argsort(pdf_vardens3)
        b = np.flipud(b)
        threshold_for_sampling = pdf_vardens3[b[int(np.rint(undersampling_ratio * len(b)))]]
        pdf_vardens4 = np.zeros((n1, n2))
        pdf_vardens4[pdf_vardens2 >= threshold_for_sampling] = 1
        sampling_mat[: n1, : n2, i] = (pdf_vardens4 > 0.1)

    return sampling_mat


def genrate_ylines_sampling_map(n1, n2, undersampling_ratio, n3):
    sampling_mat = np.zeros((n1, n2, n3), dtype=int)
    pdf_vardens_cut, _ = genPDF(n1, 9, undersampling_ratio)
    for i in range(n3):
        r_mat = np.random.rand(n1)
        pdf_vardens2 = r_mat * pdf_vardens_cut
        b = np.argsort(pdf_vardens2)
        b = np.flipud(b)
        threshold_for_sampling = pdf_vardens2[b[int(np.rint(undersampling_ratio * len(b)))]]
        pdf_vardens3 = np.zeros(n1)
        pdf_vardens3[pdf_vardens2 >= threshold_for_sampling] = 1
        sampling_mat[:, :, i] = np.tile((pdf_vardens3 > 0.1)[:, np.newaxis], (1, n2))

    return sampling_mat


def strucrand(n1, n2, n3, line, disp=0):
    Samp = np.zeros((n1, n2, n3))
    for frameno in range(n3):
        y = np.ceil(np.arange(-n1 / 2, n1 / 2, 1))
        x = np.ceil(np.linspace(-n2 / 2, n2 / 2, len(y) + 1))
        x = np.delete(x, -1)

        pi = np.pi
        inc = 0 + (pi / line) * np.random.rand(1)
        kloc_all = []
        for ang in np.arange(0, pi, pi / line):  # 0:pi / line: pi - 1e-3
            klocn = y * np.cos(ang + inc) + x * np.sin(ang + inc) * 1j
            kloc_all.append(klocn)
        kloc_all = np.array(kloc_all).T
        kcart = np.floor(kloc_all.real) + 1j * np.floor(kloc_all.imag)
        if disp:
            plt.figure()
            plt.axis('equal')
            plt.plot(kcart.real, kcart.imag)
            plt.title('k locations after nearest neighbor interpolation: Center (0,0)')
        kloc1 = np.floor(kcart.real + np.floor(n1 / 2)) + 1j * np.floor(kcart.imag + np.floor(n2 / 2))
        kloc1real = np.real(kloc1)
        kloc1real = kloc1real - n1 * (kloc1real > n1 - 1)
        kloc1imag = np.imag(kloc1)
        kloc1imag = kloc1imag - n2 * (kloc1imag > n2 - 1)
        kloc1real = kloc1real + n1 * (kloc1real < 0)
        kloc1imag = kloc1imag + n2 * (kloc1imag < 0)
        kloc1 = kloc1real + 1j * kloc1imag
        for i in range(kloc1.shape[0]):  # 1:size(kloc1, 1)
            for j in range(kloc1.shape[1]):  # 1:size(kloc1, 2)
                Samp[int(kloc1[i, j].real), int(kloc1[i, j].imag), frameno] = 1

    return Samp

def vista_mask(acc):
    f_n = 'VISTA'+str(int(acc))+'.mat'
    samp = np.array(scio.loadmat(f_n)['samp'])[...,np.newaxis]
    mask = np.transpose(np.tile(samp,(1,1,112)),[0,2,1])

    return mask
