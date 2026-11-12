function make_reference()
% Regenerate matlab_ref_128kHz_10_300.mat for util/tests/test_asd.py.
%
% Run from this folder (MATLAB R2025a or newer):
%     matlab -batch "make_reference"
%
% Then convert the result into the committed .npz with:
%     python -c "import numpy as n, scipy.io as s; \
%       r = s.loadmat('matlab_ref_128kHz_10_300.mat'); \
%       n.savez_compressed('matlab_ref_128kHz_10_300.npz', \
%         freq=r['Freq'].ravel(), psd=r['Psd'].ravel(), \
%         nf_db=float(r['nf']), fs=128000., fl=10., fh=300., \
%         nfft=131072, n=int(r['N'].ravel()[0]))"
%
% This mirrors bitstreamPSDfn_analogcl(x, 10, 300) exactly (same pwelch call,
% same WinBH7 window, same noise-floor expression), with the plotting removed.

    here = fileparts(mfilename('fullpath'));

    S = load(fullfile(here, 'asd_test_data_128kHz_10_300.mat'));
    x = S.chopout4(:);

    N  = numel(x);
    fs = 128000;
    Nfft = 0.512*2*fs;
    fl = 10; fh = 300;

    if N < Nfft
        [Psd, Freq] = periodogram(x, WinBH7(N), N, fs);
    else
        [Psd, Freq] = pwelch(x, WinBH7(Nfft), Nfft/2, Nfft, fs);
    end

    nf = 10*log10(mean(Psd(Freq>=fl & Freq<=fh)));

    out.Freq = Freq;
    out.Psd  = Psd;
    out.nf   = nf;
    out.N    = N;
    out.Nfft = Nfft;
    out.fs   = fs;
    out.fl   = fl;
    out.fh   = fh;
    save(fullfile(here, 'matlab_ref_128kHz_10_300.mat'), '-v7', '-struct', 'out');

    fprintf('N=%d Nfft=%d bins=%d df=%.10f nf=%.12f\n', ...
        N, Nfft, numel(Freq), Freq(2)-Freq(1), nf);
end

function W = WinBH7(N)
% 7-term Blackman-Harris window, verbatim from bitstreamPSDfn_analogcl.m.
    a0=0.61640321314050;
    a1=0.98537119272586;
    a2=0.49603771622007;
    a3=0.14992232793243;
    a4=0.02458719103474;
    a5=0.0017660465148687;
    a6=0.000031581188567106;

    v = 2*pi/N;
    ix = 0:N/2-1;
    m  = (ix+0.5);

    t1=cos(v*m);
    t2=cos(2*v*m);
    t3=cos(3*v*m);
    t4=cos(4*v*m);
    t5=cos(5*v*m);
    t6=cos(6*v*m);

    z = a0-a1*t1+a2*t2-a3*t3+a4*t4-a5*t5+a6*t6;

    if rem(N,2)==0
        W=[z fliplr(z)];
    else
        zflr=fliplr(z);
        W=[z zflr(2:N/2)];
    end

    W = W(:);
end
