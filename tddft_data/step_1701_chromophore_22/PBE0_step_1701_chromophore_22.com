%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1701_chromophore_22 TDDFT with PBE1PBE functional

0 1
Mg   8.635   48.046   26.058
C   6.576   48.040   28.835
C   11.350   48.445   28.327
C   10.781   48.752   23.441
C   5.937   48.335   24.010
N   8.882   48.194   28.371
C   7.918   48.187   29.284
C   8.429   48.382   30.724
C   9.965   48.753   30.457
C   10.086   48.396   28.957
C   10.160   50.185   30.785
C   8.199   47.119   31.609
C   9.233   46.843   32.710
C   8.914   45.824   33.740
O   8.828   44.645   33.528
O   8.736   46.391   35.003
N   10.791   48.543   25.858
C   11.728   48.477   26.900
C   13.046   48.403   26.365
C   12.861   48.558   24.935
C   11.415   48.713   24.709
C   14.404   48.331   27.163
C   13.904   48.498   23.743
O   13.550   48.724   22.586
C   15.346   48.113   23.993
N   8.447   48.392   23.995
C   9.410   48.684   23.145
C   8.877   48.746   21.723
C   7.319   48.603   21.880
C   7.186   48.379   23.380
C   9.299   50.007   20.890
C   6.708   47.383   21.159
C   5.259   47.671   20.744
N   6.733   48.032   26.289
C   5.710   48.202   25.392
C   4.477   48.033   25.995
C   4.759   47.924   27.397
C   6.128   47.985   27.523
C   3.195   48.178   25.287
C   4.116   47.927   28.668
O   2.904   48.013   28.900
C   5.217   47.980   29.642
C   5.062   46.991   30.715
O   5.343   45.801   30.715
O   4.617   47.613   31.795
C   4.576   46.748   33.022
C   8.872   45.452   36.049
C   9.824   46.195   36.941
C   10.127   45.944   38.198
C   9.688   44.766   38.940
C   11.192   46.836   38.817
C   10.578   47.835   39.758
C   11.392   48.306   40.972
C   10.843   47.864   42.446
C   9.750   48.912   42.916
C   11.932   47.782   43.476
C   12.575   46.398   43.619
C   12.438   45.841   45.046
C   13.402   44.632   45.226
C   14.625   44.926   46.057
C   12.795   43.223   45.562
C   13.509   42.064   44.888
C   12.945   41.566   43.450
C   13.902   42.135   42.352
C   13.224   42.374   41.011
C   15.091   41.092   42.341
H   12.200   48.645   28.982
H   11.335   48.792   22.500
H   5.062   48.578   23.403
H   7.914   49.253   31.130
H   10.764   48.208   30.960
H   9.361   50.809   31.188
H   10.394   50.719   29.864
H   10.997   50.307   31.472
H   8.025   46.240   30.988
H   7.229   47.162   32.106
H   9.615   47.729   33.217
H   10.052   46.382   32.158
H   14.860   49.301   26.967
H   15.012   47.452   26.946
H   14.189   48.406   28.229
H   15.727   48.836   24.714
H   15.925   48.105   23.070
H   15.334   47.163   24.528
H   9.293   47.880   21.208
H   6.942   49.578   21.570
H   9.996   50.555   21.524
H   8.465   50.600   20.516
H   9.832   49.607   20.028
H   6.759   46.501   21.797
H   7.224   47.111   20.239
H   5.175   47.634   19.658
H   4.974   48.687   21.018
H   4.564   47.006   21.256
H   2.863   47.383   24.619
H   3.158   49.101   24.708
H   2.428   48.178   26.062
H   5.010   48.973   30.042
H   3.536   46.426   33.077
H   4.825   47.428   33.836
H   5.300   45.936   33.088
H   9.333   44.492   35.816
H   7.912   45.294   36.541
H   10.144   47.155   36.535
H   10.533   44.087   39.054
H   8.874   44.244   38.437
H   9.245   45.171   39.850
H   11.866   47.329   38.116
H   11.839   46.135   39.345
H   9.756   47.291   40.222
H   10.247   48.709   39.197
H   11.331   49.394   40.988
H   12.438   48.032   40.839
H   10.293   46.924   42.413
H   8.786   48.411   43.006
H   9.562   49.687   42.173
H   10.111   49.376   43.834
H   11.572   48.209   44.413
H   12.751   48.452   43.216
H   13.662   46.468   43.586
H   12.318   45.682   42.838
H   11.391   45.544   45.123
H   12.665   46.643   45.748
H   13.873   44.452   44.260
H   14.693   45.982   46.317
H   15.512   44.596   45.516
H   14.604   44.349   46.982
H   11.758   43.177   45.230
H   12.783   43.134   46.649
H   13.567   41.101   45.397
H   14.556   42.349   44.779
H   11.975   42.034   43.286
H   12.723   40.502   43.366
H   14.355   43.098   42.585
H   13.034   43.429   40.813
H   12.317   41.774   40.946
H   13.893   41.956   40.258
H   15.990   41.706   42.397
H   15.147   40.538   41.404
H   15.034   40.333   43.121

