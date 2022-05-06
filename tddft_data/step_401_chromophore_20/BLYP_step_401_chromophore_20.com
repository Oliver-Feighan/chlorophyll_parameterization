%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_401_chromophore_20 TDDFT with blyp functional

0 1
Mg   6.701   56.636   40.983
C   5.504   53.539   40.969
C   9.604   55.596   39.411
C   7.546   59.907   40.255
C   3.647   57.782   42.399
N   7.509   54.790   40.425
C   6.817   53.576   40.452
C   7.616   52.456   39.864
C   9.046   53.067   39.849
C   8.731   54.582   39.866
C   10.017   52.721   41.142
C   7.046   52.095   38.419
C   7.434   50.661   38.064
C   7.616   50.315   36.528
O   8.577   49.907   35.946
O   6.425   50.427   35.906
N   8.342   57.610   39.986
C   9.486   56.997   39.468
C   10.329   57.947   38.854
C   9.809   59.251   39.174
C   8.493   58.952   39.803
C   11.623   57.623   38.191
C   10.479   60.639   39.015
O   11.536   60.744   38.409
C   9.895   61.885   39.522
N   5.576   58.502   41.091
C   6.273   59.735   40.872
C   5.453   60.953   41.275
C   4.185   60.258   41.949
C   4.461   58.754   41.796
C   6.177   62.071   42.108
C   2.840   60.796   41.463
C   1.768   60.949   42.672
N   4.936   55.798   41.718
C   3.825   56.374   42.234
C   2.896   55.383   42.639
C   3.566   54.202   42.240
C   4.700   54.544   41.537
C   1.503   55.633   43.190
C   3.482   52.724   42.118
O   2.578   51.941   42.470
C   4.688   52.251   41.262
C   4.187   51.646   40.049
O   3.400   52.185   39.342
O   4.648   50.403   39.893
C   4.097   49.825   38.682
C   6.148   49.587   34.701
C   6.624   50.317   33.491
C   6.151   50.361   32.242
C   4.973   49.602   31.781
C   6.784   51.322   31.210
C   6.570   52.823   31.447
C   5.728   53.563   30.360
C   5.763   55.109   30.623
C   4.296   55.597   30.797
C   6.505   55.845   29.385
C   8.087   55.619   29.491
C   8.754   56.758   28.722
C   9.662   57.627   29.675
C   9.093   59.071   29.852
C   11.199   57.671   29.420
C   11.999   56.542   30.158
C   12.951   55.744   29.190
C   14.161   56.661   28.812
C   14.821   56.305   27.471
C   15.159   56.592   29.948
H   10.479   55.119   38.964
H   7.779   60.955   40.052
H   2.768   58.229   42.867
H   7.482   51.535   40.431
H   9.538   52.807   38.911
H   9.590   51.859   41.655
H   10.150   53.588   41.788
H   11.008   52.576   40.713
H   7.456   52.736   37.638
H   5.959   52.173   38.410
H   6.648   49.994   38.419
H   8.322   50.311   38.590
H   12.489   58.162   38.575
H   11.556   58.014   37.176
H   11.772   56.552   38.062
H   9.548   61.798   40.551
H   9.071   62.149   38.858
H   10.653   62.668   39.489
H   5.165   61.407   40.327
H   4.361   60.386   43.017
H   7.228   61.895   42.335
H   5.627   62.245   43.034
H   6.108   63.058   41.651
H   2.470   60.149   40.668
H   2.832   61.809   41.060
H   1.423   61.955   42.910
H   2.182   60.516   43.583
H   0.929   60.273   42.506
H   1.005   56.425   42.630
H   1.521   55.976   44.224
H   0.910   54.718   43.182
H   5.356   51.496   41.676
H   4.643   48.894   38.528
H   4.175   50.430   37.779
H   3.050   49.600   38.885
H   5.064   49.559   34.818
H   6.699   48.658   34.848
H   7.354   51.068   33.792
H   4.345   49.389   32.646
H   5.077   48.705   31.171
H   4.289   50.233   31.213
H   6.255   51.032   30.302
H   7.820   50.997   31.126
H   7.562   53.266   31.535
H   6.035   52.899   32.393
H   4.706   53.184   30.366
H   6.283   53.425   29.431
H   6.291   55.429   31.521
H   3.871   55.967   29.863
H   4.171   56.284   31.634
H   3.549   54.870   31.117
H   6.038   56.808   29.590
H   6.125   55.480   28.430
H   8.351   54.713   28.946
H   8.504   55.588   30.498
H   8.089   57.406   28.150
H   9.379   56.263   27.979
H   9.622   57.162   30.661
H   9.688   59.914   29.501
H   8.940   59.248   30.917
H   8.180   59.289   29.299
H   11.591   58.562   29.910
H   11.439   57.846   28.371
H   11.289   55.750   30.395
H   12.407   56.982   31.069
H   12.443   55.524   28.251
H   13.346   54.886   29.733
H   13.893   57.711   28.695
H   14.676   57.126   26.769
H   14.576   55.287   27.169
H   15.909   56.372   27.503
H   15.497   55.615   30.294
H   14.727   57.274   30.680
H   16.042   57.198   29.750

