%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_301_chromophore_12 TDDFT with blyp functional

0 1
Mg   47.533   15.280   27.618
C   45.362   15.175   30.128
C   49.816   17.074   29.484
C   49.742   15.402   24.929
C   45.142   13.569   25.547
N   47.519   16.221   29.534
C   46.538   15.897   30.495
C   47.009   16.365   31.876
C   48.205   17.273   31.538
C   48.601   16.761   30.102
C   47.985   18.736   31.366
C   47.194   15.307   33.054
C   47.168   15.876   34.456
C   48.317   15.535   35.357
O   48.807   14.442   35.589
O   48.832   16.622   36.038
N   49.500   16.178   27.242
C   50.265   16.866   28.201
C   51.496   17.271   27.628
C   51.486   16.903   26.282
C   50.227   16.057   26.105
C   52.629   18.070   28.372
C   52.541   17.087   25.224
O   52.461   16.446   24.211
C   53.772   17.957   25.342
N   47.429   14.689   25.522
C   48.529   14.807   24.640
C   48.266   14.068   23.314
C   46.876   13.351   23.640
C   46.376   13.971   24.905
C   48.203   15.058   22.158
C   46.946   11.837   23.756
C   47.818   11.050   22.779
N   45.595   14.640   27.692
C   44.788   13.928   26.907
C   43.584   13.678   27.534
C   43.767   14.129   28.820
C   45.036   14.669   28.881
C   42.427   12.908   27.029
C   43.204   14.173   30.135
O   42.113   13.829   30.503
C   44.216   14.803   31.037
C   43.714   15.930   31.836
O   43.484   17.067   31.424
O   43.657   15.549   33.164
C   43.107   16.415   34.183
C   49.555   16.447   37.275
C   49.545   17.644   38.147
C   50.565   18.221   38.821
C   51.900   17.429   39.202
C   50.375   19.633   39.389
C   51.762   20.368   39.438
C   51.600   21.974   39.350
C   52.754   22.680   38.520
C   54.084   22.852   39.372
C   52.363   24.036   37.845
C   52.757   23.989   36.342
C   51.657   24.726   35.537
C   52.090   24.782   34.047
C   52.812   26.095   33.679
C   50.888   24.495   33.077
C   51.505   24.140   31.677
C   50.579   24.661   30.514
C   50.561   23.642   29.365
C   50.976   24.505   28.110
C   49.192   22.814   29.148
H   50.493   17.545   30.199
H   50.356   15.389   24.026
H   44.426   12.930   25.026
H   46.219   17.025   32.234
H   49.111   17.043   32.098
H   47.742   19.002   30.338
H   48.836   19.350   31.665
H   47.134   19.008   31.991
H   48.138   14.778   32.921
H   46.504   14.524   32.738
H   46.263   15.447   34.886
H   47.006   16.925   34.209
H   52.793   19.050   27.925
H   53.485   17.396   28.367
H   52.512   18.266   29.438
H   53.602   18.993   25.633
H   54.158   17.936   24.323
H   54.473   17.526   26.058
H   49.092   13.361   23.231
H   46.201   13.673   22.847
H   47.188   15.362   21.903
H   48.661   14.613   21.275
H   48.782   15.954   22.380
H   45.928   11.447   23.737
H   47.421   11.586   24.704
H   48.660   10.557   23.264
H   48.330   11.648   22.024
H   47.265   10.240   22.305
H   42.403   12.811   25.943
H   41.594   13.385   27.545
H   42.462   11.961   27.568
H   44.530   13.999   31.704
H   43.862   16.857   34.833
H   42.437   15.805   34.790
H   42.505   17.198   33.722
H   50.571   16.336   36.895
H   49.253   15.597   37.887
H   48.688   18.300   37.994
H   52.173   17.893   40.150
H   52.693   17.510   38.459
H   51.713   16.355   39.228
H   49.835   19.629   40.336
H   49.872   20.164   38.581
H   52.359   19.997   38.605
H   52.267   20.256   40.397
H   51.591   22.393   40.356
H   50.726   22.250   38.758
H   53.181   21.965   37.818
H   54.857   23.435   38.871
H   54.519   21.875   39.579
H   53.812   23.439   40.249
H   52.967   24.866   38.213
H   51.292   24.190   37.975
H   52.839   22.985   35.926
H   53.684   24.525   36.138
H   51.767   25.728   35.953
H   50.664   24.319   35.727
H   52.809   23.978   33.891
H   52.215   26.667   32.969
H   53.793   25.979   33.219
H   52.958   26.647   34.608
H   50.205   25.338   32.969
H   50.301   23.605   33.306
H   51.496   23.050   31.686
H   52.526   24.523   31.699
H   50.948   25.636   30.197
H   49.529   24.743   30.794
H   51.325   22.878   29.508
H   50.918   23.919   27.193
H   51.841   25.159   28.221
H   50.130   25.193   28.076
H   49.193   21.823   28.695
H   48.541   23.435   28.533
H   48.722   22.742   30.129

