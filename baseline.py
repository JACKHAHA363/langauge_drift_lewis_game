"""
Baseline selfplay
"""
from drift.lewis.gumbel import selfplay
from drift.lewis.linear import Speaker, Listener

if __name__ == '__main__':
    speaker = Speaker.load('s_sl.pth')
    listener = Listener.load('l_sl.pth')
    stats, speaker, listener = selfplay(speaker, listener)
    logstr = []
    for name, val in stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
    print(' '.join(logstr))
