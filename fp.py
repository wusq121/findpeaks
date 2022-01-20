import numpy as np
from scipy.ndimage import gaussian_filter1d


def featurepoint(x, l, show=False):
    x = gaussian_filter1d(x, 10, mode='constant')
    dx = x[1:] - x[: -1]
    dx = np.hstack((0, dx))
    idx = get_peaks(np.abs(dx), mpd=l, show=show)
    return idx


def get_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')  # 转换为至少1维数据，主要是将标量进行转换。
    if x.size < 3:  # 如果数据量少于三个，则没有peak，返回空.
        return np.array([], dtype=np.int_)
    if valley:  # 如果找波谷，则把数据球翻转，求反的波峰
        x = -x

    dx = x[1:] - x[:-1]

    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))][0] = np.inf

    ine, ire, ife = np.array([[], [], []], dtype=np.int_)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[
            0]  # 这是一阶差分中穿越0点的值，象征x中的极大值点
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    idx = np.unique(np.hstack((ine, ire, ife)))

    thx = np.median(x[idx])
    y = np.hstack((0, dx)) - np.hstack((dx, 0))
    thy = np.median(y[idx])
    ind = []
    for i in idx:
        if i > x.size - mpd:
            continue
        if x[i] > thx and y[i] > thy:
            ind.append(i)
    ind = np.array(ind, dtype=np.int_)

    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(
            np.hstack((indnan, indnan-1, indnan+1))), invert=True)]

    if ind.size and ind[0] == 0:
        ind = ind[1: 0]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[: -1]

    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind] - x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])

    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # 按照大小排序
        idel = np.zeros(ind.size, dtype=np.bool_)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | (
                    ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2,
                    ms=8, label='{} {}'.format(ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin-0.1*yrange, ymax+0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title(
            f'{mode} (mph={mph}, mpd={mpd}, threshold={threshold}, edge={edge})')
        plt.show()
