%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_401_chromophore_10 TDDFT with cam-b3lyp functional

0 1
Mg   40.925   7.661   28.749
C   42.591   8.942   31.553
C   38.457   6.150   30.801
C   39.231   6.601   26.001
C   43.627   8.601   26.743
N   40.653   7.501   30.927
C   41.385   8.220   31.846
C   40.737   8.050   33.174
C   39.554   6.963   32.974
C   39.558   6.797   31.453
C   39.921   5.500   33.564
C   40.190   9.398   33.810
C   40.893   9.854   35.127
C   40.276   10.719   36.177
O   39.185   11.237   36.160
O   41.299   10.944   37.096
N   39.126   6.560   28.424
C   38.238   6.062   29.442
C   37.029   5.410   28.768
C   37.192   5.745   27.397
C   38.522   6.377   27.217
C   36.017   4.608   29.535
C   36.235   5.409   26.365
O   36.361   5.713   25.198
C   34.903   4.868   26.750
N   41.295   7.712   26.697
C   40.467   7.148   25.713
C   41.177   7.026   24.307
C   42.566   7.709   24.557
C   42.537   8.016   26.094
C   41.224   5.535   23.763
C   42.876   8.964   23.661
C   42.001   10.175   24.092
N   42.713   8.583   29.059
C   43.695   8.891   28.181
C   44.820   9.500   28.898
C   44.332   9.592   30.184
C   43.100   8.983   30.240
C   46.118   9.966   28.342
C   44.688   10.073   31.542
O   45.672   10.626   31.892
C   43.503   9.691   32.536
C   44.049   8.957   33.636
O   44.001   7.743   33.695
O   44.418   9.893   34.571
C   44.822   9.221   35.800
C   41.098   11.828   38.252
C   40.997   11.146   39.548
C   41.021   11.689   40.779
C   41.121   13.196   41.038
C   41.276   10.746   41.971
C   40.462   11.030   43.221
C   39.919   9.754   43.795
C   38.429   9.780   44.147
C   37.952   8.345   44.723
C   37.579   10.225   42.944
C   36.865   11.574   43.109
C   35.378   11.500   42.774
C   34.361   11.699   44.027
C   33.048   10.931   43.754
C   34.067   13.060   44.578
C   33.524   13.057   46.016
C   34.511   13.616   47.153
C   35.200   12.565   48.056
C   34.038   12.050   48.917
C   36.448   13.079   48.735
H   37.686   5.827   31.504
H   38.681   6.251   25.124
H   44.488   8.856   26.122
H   41.422   7.425   33.747
H   38.618   7.261   33.446
H   40.203   4.896   32.701
H   39.039   5.105   34.069
H   40.617   5.604   34.395
H   39.125   9.251   33.985
H   40.331   10.222   33.110
H   41.767   10.435   34.835
H   41.297   8.957   35.596
H   36.429   4.567   30.544
H   35.848   3.675   28.999
H   35.051   5.108   29.596
H   34.353   5.628   27.304
H   34.974   3.928   27.297
H   34.306   4.683   25.857
H   40.607   7.514   23.516
H   43.264   6.891   24.376
H   42.201   5.075   23.914
H   41.016   5.390   22.703
H   40.434   4.946   24.229
H   42.732   8.731   22.607
H   43.919   9.215   23.858
H   42.628   11.031   24.341
H   41.219   9.924   24.809
H   41.591   10.439   23.117
H   46.046   10.917   27.814
H   46.580   9.200   27.718
H   46.769   10.014   29.215
H   42.914   10.589   32.722
H   43.937   8.750   36.228
H   45.135   9.935   36.561
H   45.547   8.408   35.744
H   40.275   12.518   38.065
H   42.038   12.380   38.272
H   40.883   10.063   39.496
H   42.144   13.476   40.786
H   41.037   13.491   42.083
H   40.375   13.748   40.465
H   42.323   10.810   42.268
H   41.177   9.715   41.632
H   39.779   11.867   43.073
H   41.196   11.364   43.954
H   40.578   9.656   44.658
H   40.258   8.882   43.236
H   38.262   10.468   44.975
H   37.019   8.062   44.237
H   37.824   8.544   45.787
H   38.759   7.614   44.670
H   36.804   9.495   42.707
H   38.241   10.282   42.080
H   37.403   12.196   42.394
H   36.959   12.013   44.102
H   35.089   10.741   42.046
H   35.199   12.426   42.228
H   34.883   11.090   44.765
H   33.256   9.932   43.369
H   32.469   11.412   42.966
H   32.485   10.836   44.682
H   33.178   13.393   44.043
H   34.854   13.799   44.430
H   33.355   12.010   46.268
H   32.520   13.479   46.069
H   33.874   14.319   47.689
H   35.276   14.151   46.589
H   35.380   11.699   47.419
H   34.545   11.408   49.638
H   33.445   11.367   48.310
H   33.486   12.902   49.315
H   36.409   13.042   49.823
H   36.745   14.082   48.426
H   37.213   12.413   48.336

