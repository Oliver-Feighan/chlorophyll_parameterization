%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1051_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   15.975   51.301   25.625
C   17.765   50.277   28.382
C   13.524   52.415   27.669
C   14.279   52.179   22.937
C   18.563   50.052   23.524
N   15.779   51.358   27.808
C   16.612   51.031   28.748
C   16.122   51.339   30.132
C   14.600   51.559   29.896
C   14.584   51.764   28.376
C   13.630   50.479   30.429
C   16.838   52.527   30.784
C   17.252   52.291   32.242
C   16.244   51.513   33.148
O   16.476   50.483   33.807
O   14.984   52.157   33.157
N   14.175   52.330   25.380
C   13.303   52.676   26.330
C   12.161   53.369   25.717
C   12.318   53.201   24.329
C   13.630   52.496   24.179
C   11.097   54.025   26.580
C   11.453   53.524   23.195
O   11.753   53.298   22.058
C   10.196   54.234   23.462
N   16.328   50.999   23.432
C   15.520   51.681   22.634
C   15.994   51.450   21.132
C   17.377   50.708   21.327
C   17.487   50.629   22.827
C   15.130   50.836   20.035
C   18.662   51.413   20.600
C   19.301   50.749   19.435
N   17.667   50.148   25.828
C   18.621   49.716   24.896
C   19.679   49.009   25.654
C   19.362   49.197   27.065
C   18.097   49.853   27.078
C   20.924   48.396   25.003
C   19.895   49.061   28.443
O   20.886   48.633   28.965
C   18.748   49.636   29.281
C   18.196   48.425   29.992
O   17.950   47.327   29.552
O   18.099   48.721   31.327
C   17.878   47.643   32.282
C   14.080   51.688   34.227
C   12.642   52.230   34.081
C   11.547   51.972   34.820
C   11.528   51.053   36.027
C   10.142   52.634   34.569
C   9.721   53.840   35.426
C   9.357   55.091   34.550
C   7.927   55.660   34.826
C   7.627   56.762   33.836
C   8.049   56.316   36.326
C   6.857   56.120   37.231
C   6.503   57.362   37.965
C   5.166   57.991   37.589
C   5.185   59.502   38.008
C   3.979   57.352   38.397
C   2.692   57.365   37.552
C   1.745   58.519   37.958
C   0.378   58.090   38.392
C   0.434   57.344   39.738
C   -0.454   57.368   37.317
H   12.733   52.837   28.292
H   13.782   52.444   22.002
H   19.432   49.638   23.008
H   16.229   50.404   30.681
H   14.310   52.462   30.434
H   14.017   49.868   31.244
H   13.389   49.857   29.567
H   12.712   50.908   30.831
H   16.299   53.474   30.819
H   17.716   52.798   30.199
H   17.544   53.243   32.684
H   18.146   51.669   32.188
H   11.309   54.089   27.647
H   10.143   53.513   26.449
H   10.989   55.045   26.213
H   10.426   55.065   24.129
H   9.451   53.627   23.978
H   9.880   54.683   22.520
H   16.144   52.460   20.751
H   17.336   49.687   20.946
H   14.666   51.535   19.339
H   14.486   50.015   20.348
H   15.895   50.342   19.435
H   19.454   51.607   21.323
H   18.326   52.369   20.197
H   18.864   51.057   18.485
H   19.162   49.676   19.566
H   20.346   51.018   19.280
H   21.588   48.548   25.854
H   21.325   48.934   24.144
H   20.798   47.322   24.863
H   19.206   50.269   30.041
H   17.459   48.034   33.210
H   18.807   47.135   32.539
H   17.067   47.023   31.899
H   14.516   52.101   35.137
H   14.057   50.605   34.349
H   12.463   52.858   33.209
H   10.984   51.544   36.835
H   12.538   50.820   36.363
H   10.974   50.154   35.758
H   9.387   51.857   34.686
H   9.979   52.929   33.532
H   10.451   54.236   36.132
H   8.987   53.561   36.181
H   9.356   54.897   33.477
H   10.124   55.851   34.698
H   7.181   54.867   34.879
H   8.488   57.424   33.744
H   6.846   57.400   34.250
H   7.291   56.192   32.969
H   8.393   57.347   36.241
H   8.831   55.826   36.906
H   7.172   55.294   37.869
H   6.035   55.730   36.630
H   7.334   58.068   37.964
H   6.557   57.062   39.012
H   4.996   58.001   36.513
H   4.184   59.933   38.007
H   5.506   60.158   37.199
H   5.580   59.738   38.996
H   3.807   57.881   39.334
H   4.228   56.328   38.674
H   2.217   56.385   37.588
H   3.066   57.584   36.551
H   1.653   59.232   37.139
H   2.186   59.079   38.783
H   -0.104   59.038   38.630
H   1.458   57.095   40.016
H   -0.184   56.451   39.650
H   -0.006   57.926   40.548
H   -1.337   57.916   36.987
H   -0.918   56.481   37.749
H   0.184   57.084   36.481

