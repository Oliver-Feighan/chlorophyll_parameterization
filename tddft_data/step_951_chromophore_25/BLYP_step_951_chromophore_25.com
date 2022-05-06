%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_951_chromophore_25 TDDFT with blyp functional

0 1
Mg   -2.534   34.509   26.894
C   -3.425   32.595   29.875
C   -1.279   36.824   28.935
C   -2.121   36.412   24.055
C   -4.073   32.041   25.077
N   -2.395   34.632   29.063
C   -2.754   33.785   30.152
C   -2.324   34.285   31.549
C   -1.625   35.637   31.132
C   -1.755   35.685   29.607
C   -2.277   36.853   31.855
C   -1.401   33.210   32.287
C   -1.975   32.771   33.618
C   -1.029   32.823   34.783
O   -0.177   31.999   35.110
O   -1.143   34.064   35.405
N   -1.726   36.326   26.514
C   -1.292   37.196   27.497
C   -0.779   38.415   26.961
C   -1.214   38.322   25.563
C   -1.725   37.013   25.289
C   -0.017   39.485   27.678
C   -1.079   39.379   24.469
O   -1.439   39.189   23.278
C   -0.499   40.761   24.885
N   -3.114   34.319   24.942
C   -2.704   35.163   23.875
C   -3.286   34.746   22.546
C   -3.600   33.191   22.865
C   -3.567   33.151   24.395
C   -4.510   35.592   22.043
C   -2.657   32.047   22.272
C   -1.328   31.617   23.001
N   -3.545   32.725   27.315
C   -4.152   31.799   26.470
C   -4.601   30.667   27.163
C   -4.455   30.968   28.524
C   -3.728   32.196   28.538
C   -5.442   29.542   26.652
C   -4.675   30.516   29.940
O   -5.264   29.542   30.308
C   -3.946   31.481   30.885
C   -4.931   31.997   31.921
O   -5.877   32.716   31.731
O   -4.611   31.473   33.159
C   -5.633   31.857   34.114
C   -0.264   34.268   36.565
C   -0.907   35.201   37.534
C   -0.307   35.896   38.526
C   1.076   35.723   39.059
C   -1.073   36.906   39.334
C   -1.674   36.432   40.622
C   -1.361   37.353   41.904
C   -2.458   37.171   42.984
C   -3.682   38.047   42.586
C   -1.962   37.415   44.457
C   -1.687   36.135   45.258
C   -0.740   36.443   46.433
C   0.628   35.693   46.613
C   0.941   35.470   48.021
C   1.845   36.487   45.957
C   2.133   36.006   44.524
C   3.216   34.936   44.436
C   4.347   35.317   43.416
C   4.545   34.203   42.404
C   5.667   35.750   44.022
H   -0.776   37.500   29.629
H   -1.998   37.030   23.164
H   -4.209   31.105   24.531
H   -3.227   34.455   32.136
H   -0.568   35.642   31.400
H   -3.153   36.598   32.452
H   -2.574   37.541   31.063
H   -1.512   37.218   32.541
H   -0.401   33.627   32.402
H   -1.285   32.370   31.604
H   -2.269   31.731   33.475
H   -2.846   33.389   33.836
H   -0.605   40.380   27.476
H   1.001   39.588   27.301
H   -0.035   39.373   28.762
H   -1.202   41.367   25.457
H   -0.352   41.260   23.927
H   0.491   40.806   25.338
H   -2.607   34.697   21.694
H   -4.647   32.999   22.630
H   -4.186   36.105   21.137
H   -4.694   36.375   22.778
H   -5.437   35.046   21.871
H   -2.278   32.576   21.398
H   -3.299   31.236   21.928
H   -0.522   31.822   22.295
H   -1.447   30.564   23.256
H   -1.261   32.222   23.905
H   -5.138   29.191   25.666
H   -6.469   29.908   26.670
H   -5.431   28.731   27.380
H   -3.097   31.041   31.409
H   -5.908   32.910   34.045
H   -5.193   31.608   35.080
H   -6.503   31.201   34.087
H   0.769   34.561   36.377
H   -0.112   33.377   37.174
H   -1.909   35.442   37.177
H   1.109   35.507   40.127
H   1.649   36.642   38.935
H   1.497   34.863   38.537
H   -1.879   37.339   38.742
H   -0.360   37.680   39.619
H   -1.247   35.468   40.896
H   -2.739   36.207   40.567
H   -1.285   38.379   41.544
H   -0.367   37.035   42.219
H   -2.796   36.134   43.009
H   -3.531   38.583   41.649
H   -4.000   38.718   43.384
H   -4.597   37.470   42.451
H   -2.850   37.790   44.966
H   -1.143   38.133   44.423
H   -1.207   35.453   44.555
H   -2.673   35.809   45.588
H   -1.381   36.259   47.295
H   -0.507   37.508   46.423
H   0.477   34.718   46.150
H   0.695   36.325   48.650
H   1.994   35.241   48.185
H   0.375   34.662   48.484
H   2.763   36.594   46.535
H   1.483   37.489   45.726
H   2.464   36.852   43.921
H   1.162   35.827   44.064
H   2.820   33.943   44.224
H   3.626   34.930   45.446
H   4.016   36.212   42.889
H   3.599   34.057   41.882
H   4.884   33.285   42.883
H   5.351   34.378   41.691
H   5.730   35.663   45.106
H   5.749   36.833   43.929
H   6.565   35.322   43.576

