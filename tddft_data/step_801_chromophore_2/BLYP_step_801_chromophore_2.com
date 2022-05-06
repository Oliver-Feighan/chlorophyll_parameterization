%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_801_chromophore_2 TDDFT with blyp functional

0 1
Mg   3.456   0.695   44.443
C   6.255   2.778   44.255
C   1.553   3.203   42.981
C   0.899   -1.555   43.909
C   5.644   -1.958   45.289
N   3.878   2.673   43.700
C   5.114   3.307   43.664
C   4.973   4.775   43.282
C   3.360   4.951   43.206
C   2.864   3.555   43.335
C   2.699   5.897   44.260
C   5.745   5.117   41.897
C   6.088   3.970   40.933
C   6.516   4.253   39.501
O   7.192   5.179   39.132
O   6.149   3.240   38.642
N   1.431   0.790   43.704
C   0.884   1.914   43.134
C   -0.423   1.596   42.663
C   -0.646   0.224   42.976
C   0.557   -0.262   43.590
C   -1.418   2.628   42.031
C   -1.923   -0.647   42.560
O   -2.832   -0.090   41.981
C   -2.204   -2.067   43.023
N   3.324   -1.469   44.378
C   2.116   -2.127   44.289
C   2.360   -3.595   44.513
C   3.780   -3.734   45.184
C   4.308   -2.320   44.926
C   1.300   -4.421   45.167
C   4.689   -4.983   44.718
C   5.350   -5.734   45.823
N   5.501   0.352   44.673
C   6.236   -0.709   45.168
C   7.559   -0.310   45.499
C   7.691   1.005   45.079
C   6.398   1.377   44.655
C   8.642   -1.202   45.982
C   8.546   2.204   44.973
O   9.737   2.320   45.141
C   7.604   3.353   44.495
C   7.615   4.556   45.355
O   8.387   5.477   45.148
O   6.696   4.583   46.345
C   6.620   5.819   47.163
C   6.594   3.257   37.281
C   5.540   2.642   36.329
C   5.952   1.916   35.231
C   7.384   1.959   34.691
C   4.912   1.150   34.417
C   3.715   1.964   33.822
C   3.006   1.243   32.642
C   1.434   1.266   32.758
C   0.755   0.067   33.406
C   0.718   1.583   31.412
C   -0.105   2.964   31.380
C   -1.348   2.990   30.507
C   -1.406   4.104   29.499
C   -1.873   5.373   30.169
C   -2.236   3.811   28.290
C   -1.363   3.316   27.088
C   -1.475   4.347   25.919
C   -1.804   3.787   24.517
C   -1.179   4.610   23.366
C   -3.314   3.375   24.490
H   1.010   4.083   42.631
H   0.135   -2.325   43.781
H   6.273   -2.736   45.727
H   5.382   5.314   44.137
H   3.111   5.380   42.236
H   3.024   5.486   45.216
H   1.647   5.799   43.991
H   3.020   6.934   44.164
H   6.643   5.546   42.344
H   5.178   5.898   41.389
H   5.401   3.125   40.963
H   7.025   3.525   41.265
H   -1.414   2.386   40.968
H   -1.062   3.654   42.119
H   -2.432   2.428   42.377
H   -1.946   -2.187   44.075
H   -1.559   -2.593   42.319
H   -3.270   -2.291   42.979
H   2.312   -3.870   43.459
H   3.581   -3.703   46.256
H   0.644   -3.826   45.801
H   1.805   -5.229   45.696
H   0.835   -4.924   44.319
H   5.335   -4.591   43.933
H   4.092   -5.731   44.196
H   4.916   -5.430   46.776
H   6.412   -5.502   45.753
H   5.131   -6.801   45.827
H   8.888   -2.028   45.315
H   8.375   -1.533   46.985
H   9.616   -0.714   46.010
H   8.247   3.593   43.648
H   5.846   6.475   46.766
H   7.576   6.342   47.177
H   6.320   5.545   48.174
H   7.414   2.541   37.218
H   6.919   4.225   36.900
H   4.487   2.579   36.603
H   7.922   1.038   34.920
H   7.867   2.875   35.031
H   7.433   1.982   33.603
H   4.578   0.393   35.126
H   5.480   0.643   33.637
H   4.060   2.929   33.452
H   3.033   2.093   34.662
H   3.350   0.221   32.486
H   3.252   1.757   31.713
H   1.134   2.156   33.312
H   -0.126   0.376   33.969
H   1.426   -0.416   34.116
H   0.577   -0.658   32.612
H   0.191   0.718   31.010
H   1.417   1.757   30.594
H   0.677   3.698   31.187
H   -0.395   3.309   32.373
H   -2.225   3.126   31.140
H   -1.399   2.045   29.967
H   -0.421   4.449   29.183
H   -1.144   6.113   29.841
H   -1.782   5.258   31.249
H   -2.858   5.648   29.792
H   -2.901   4.647   28.070
H   -2.932   2.991   28.469
H   -1.728   2.359   26.715
H   -0.303   3.366   27.333
H   -0.491   4.809   25.845
H   -2.236   5.120   26.029
H   -1.186   2.889   24.495
H   -0.449   5.369   23.647
H   -1.830   5.106   22.646
H   -0.601   3.864   22.820
H   -3.807   3.640   23.555
H   -3.867   3.778   25.339
H   -3.237   2.295   24.618

