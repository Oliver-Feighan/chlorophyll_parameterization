%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1701_chromophore_2 TDDFT with PBE1PBE functional

0 1
Mg   3.794   -0.553   43.417
C   7.203   0.041   43.104
C   3.124   2.228   41.593
C   0.506   -1.592   43.070
C   4.526   -3.737   44.675
N   5.065   0.876   42.522
C   6.459   1.020   42.480
C   6.956   2.245   41.774
C   5.566   3.073   41.768
C   4.472   2.019   41.968
C   5.394   4.202   42.828
C   7.610   1.993   40.397
C   7.275   0.712   39.600
C   6.252   0.789   38.470
O   5.033   0.607   38.538
O   6.855   0.991   37.248
N   2.039   0.290   42.468
C   1.966   1.490   41.889
C   0.630   1.799   41.563
C   -0.234   0.738   42.029
C   0.766   -0.297   42.556
C   0.183   3.081   40.999
C   -1.757   0.658   42.022
O   -2.441   1.564   41.559
C   -2.577   -0.449   42.496
N   2.659   -2.375   43.826
C   1.349   -2.530   43.582
C   0.881   -4.003   43.959
C   2.255   -4.722   44.293
C   3.236   -3.533   44.255
C   -0.119   -4.106   45.210
C   2.679   -5.913   43.287
C   2.893   -7.193   44.052
N   5.427   -1.603   44.061
C   5.582   -2.842   44.602
C   6.994   -3.133   45.037
C   7.669   -2.041   44.454
C   6.692   -1.114   43.873
C   7.514   -4.319   45.752
C   9.020   -1.466   44.122
O   10.164   -1.800   44.374
C   8.751   -0.114   43.280
C   9.378   1.070   43.927
O   9.860   2.024   43.297
O   9.219   1.120   45.286
C   9.559   2.321   45.988
C   6.013   1.316   36.074
C   6.555   0.517   34.906
C   6.201   -0.662   34.341
C   5.089   -1.523   34.775
C   6.895   -1.125   33.076
C   6.159   -1.011   31.661
C   6.813   0.047   30.806
C   5.829   0.754   29.818
C   6.809   1.409   28.707
C   4.986   1.790   30.535
C   3.559   2.012   29.945
C   2.483   1.411   30.921
C   1.887   2.465   31.901
C   2.385   2.045   33.323
C   0.312   2.463   31.811
C   -0.256   3.691   31.093
C   -1.706   3.558   30.454
C   -2.126   4.692   29.457
C   -2.148   6.029   30.157
C   -3.461   4.314   28.758
H   3.039   3.162   41.033
H   -0.514   -1.922   43.276
H   4.664   -4.711   45.150
H   7.709   2.707   42.412
H   5.542   3.623   40.827
H   5.204   5.194   42.417
H   6.365   4.362   43.299
H   4.573   4.081   43.534
H   8.697   1.922   40.435
H   7.482   2.831   39.711
H   6.737   0.059   40.288
H   8.225   0.230   39.369
H   1.001   3.469   40.391
H   0.083   3.844   41.771
H   -0.665   3.083   40.315
H   -2.478   -0.730   43.545
H   -2.194   -1.272   41.893
H   -3.639   -0.300   42.303
H   0.490   -4.488   43.065
H   2.283   -5.092   45.318
H   0.145   -5.039   45.709
H   -1.178   -4.122   44.955
H   0.025   -3.277   45.902
H   3.497   -5.627   42.625
H   1.845   -6.068   42.602
H   3.314   -7.959   43.401
H   1.994   -7.476   44.601
H   3.683   -7.055   44.791
H   8.576   -4.357   45.991
H   7.237   -5.149   45.102
H   6.906   -4.461   46.645
H   9.274   -0.062   42.325
H   9.241   2.105   47.008
H   8.885   3.113   45.662
H   10.617   2.550   45.857
H   6.232   2.314   35.695
H   4.989   0.961   36.185
H   7.250   1.132   34.334
H   4.789   -1.438   35.819
H   5.416   -2.543   34.570
H   4.215   -1.286   34.168
H   7.279   -2.135   33.219
H   7.856   -0.617   33.001
H   5.088   -0.958   31.860
H   6.304   -2.001   31.227
H   7.738   -0.416   30.459
H   7.084   0.837   31.506
H   5.239   -0.010   29.312
H   6.740   2.497   28.714
H   6.421   1.158   27.720
H   7.856   1.135   28.833
H   5.385   2.781   30.317
H   5.010   1.604   31.608
H   3.324   1.398   29.076
H   3.326   3.022   29.606
H   2.788   0.496   31.429
H   1.670   1.053   30.290
H   2.337   3.431   31.671
H   1.636   1.489   33.886
H   2.740   2.888   33.916
H   3.228   1.356   33.273
H   -0.091   2.098   32.756
H   0.069   1.662   31.113
H   0.463   3.974   30.324
H   -0.207   4.471   31.853
H   -2.418   3.358   31.255
H   -1.593   2.664   29.841
H   -1.424   4.732   28.625
H   -1.291   6.591   29.786
H   -2.122   5.918   31.241
H   -2.972   6.657   29.819
H   -3.338   4.436   27.682
H   -4.245   5.011   29.052
H   -3.768   3.291   28.975

