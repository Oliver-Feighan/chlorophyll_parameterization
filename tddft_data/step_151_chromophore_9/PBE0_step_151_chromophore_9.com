%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   35.960   1.746   30.004
C   33.751   2.709   32.559
C   38.444   1.744   32.239
C   38.028   1.228   27.491
C   33.342   2.586   27.623
N   36.074   2.340   32.123
C   35.115   2.510   32.985
C   35.572   2.710   34.434
C   37.124   2.522   34.298
C   37.220   2.147   32.797
C   37.934   3.707   34.765
C   35.037   1.503   35.324
C   34.693   1.860   36.764
C   35.433   3.018   37.410
O   35.018   4.141   37.289
O   36.539   2.663   38.079
N   37.997   1.332   29.933
C   38.847   1.352   30.922
C   40.176   0.958   30.455
C   40.056   0.712   29.052
C   38.611   1.097   28.743
C   41.409   1.084   31.346
C   41.117   0.458   28.065
O   40.896   0.621   26.862
C   42.588   0.022   28.415
N   35.643   1.663   27.889
C   36.747   1.554   27.033
C   36.335   1.744   25.559
C   34.863   2.150   25.665
C   34.569   2.146   27.160
C   37.210   2.863   24.972
C   33.883   1.166   24.883
C   33.327   -0.013   25.733
N   33.923   2.341   30.019
C   32.948   2.616   28.996
C   31.769   3.062   29.664
C   32.021   3.073   30.999
C   33.319   2.620   31.191
C   30.543   3.564   28.977
C   31.331   3.250   32.299
O   30.181   3.586   32.596
C   32.450   2.867   33.413
C   32.564   3.979   34.465
O   33.210   5.018   34.417
O   31.617   3.695   35.473
C   31.428   4.812   36.398
C   37.166   3.695   38.827
C   38.040   2.857   39.711
C   37.733   2.159   40.810
C   36.312   1.961   41.277
C   38.811   1.387   41.616
C   39.911   2.234   42.251
C   40.617   1.514   43.540
C   41.938   0.787   43.153
C   41.725   -0.712   42.834
C   43.057   0.970   44.149
C   44.419   1.464   43.554
C   45.046   2.606   44.378
C   45.459   3.848   43.484
C   46.673   4.485   44.132
C   44.259   4.878   43.367
C   44.315   5.655   42.023
C   44.749   7.165   42.164
C   43.631   8.280   41.924
C   44.092   9.595   41.307
C   43.052   8.667   43.328
H   39.205   1.727   33.021
H   38.752   1.063   26.689
H   32.712   2.982   26.824
H   35.240   3.700   34.747
H   37.457   1.658   34.873
H   38.167   4.383   33.942
H   38.875   3.357   35.190
H   37.351   4.325   35.447
H   35.740   0.682   35.468
H   34.136   1.151   34.822
H   35.002   0.981   37.329
H   33.614   2.005   36.817
H   42.001   1.800   30.776
H   41.957   0.143   31.402
H   41.175   1.374   32.370
H   43.147   0.958   28.418
H   42.836   -0.577   27.540
H   42.885   -0.540   29.300
H   36.385   0.742   25.132
H   34.695   3.182   25.356
H   37.926   2.487   24.242
H   37.736   3.249   25.845
H   36.547   3.597   24.514
H   34.520   0.777   24.088
H   33.140   1.880   24.529
H   33.333   -0.900   25.099
H   32.292   0.236   25.970
H   33.968   -0.112   26.608
H   29.868   2.720   28.832
H   30.879   4.006   28.039
H   30.027   4.314   29.577
H   32.250   1.904   33.883
H   32.201   5.017   37.138
H   30.468   4.735   36.908
H   31.385   5.736   35.820
H   36.483   4.065   39.592
H   37.676   4.419   38.192
H   39.114   2.979   39.570
H   35.734   2.866   41.087
H   36.320   1.826   42.358
H   35.826   1.050   40.928
H   39.288   0.694   40.922
H   38.334   0.756   42.367
H   39.515   3.175   42.633
H   40.537   2.396   41.373
H   39.890   0.835   43.986
H   40.896   2.333   44.203
H   42.319   1.251   42.243
H   42.396   -1.319   43.441
H   42.009   -0.892   41.797
H   40.705   -1.067   42.978
H   43.464   0.122   44.700
H   42.791   1.772   44.838
H   44.306   1.872   42.550
H   45.077   0.595   43.528
H   45.893   2.286   44.984
H   44.368   3.078   45.089
H   45.864   3.632   42.496
H   46.390   5.256   44.848
H   47.210   5.048   43.369
H   47.399   3.751   44.483
H   44.228   5.578   44.202
H   43.391   4.227   43.477
H   43.359   5.587   41.506
H   44.979   5.269   41.250
H   45.406   7.258   41.299
H   45.370   7.265   43.055
H   42.876   7.842   41.271
H   45.167   9.677   41.467
H   43.553   10.529   41.462
H   43.924   9.409   40.246
H   43.551   8.067   44.088
H   42.039   8.265   43.334
H   42.960   9.735   43.522

