import torch


def create_window(window_size, channel):
    def gaussian(window_size_2, sigma):
        gauss = torch.exp(
            torch.tensor([-(x - window_size_2 // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size_2)]))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window
