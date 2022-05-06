%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_851_chromophore_5 TDDFT with wB97XD functional

0 1
Mg   24.449   -6.273   45.988
C   26.886   -4.089   45.060
C   22.241   -4.434   44.070
C   22.331   -8.843   46.024
C   26.938   -8.263   47.296
N   24.629   -4.681   44.465
C   25.720   -3.863   44.358
C   25.525   -2.655   43.502
C   23.964   -2.536   43.642
C   23.567   -3.970   44.105
C   23.363   -1.435   44.575
C   26.042   -3.056   42.105
C   25.850   -2.043   40.884
C   24.758   -2.280   39.752
O   23.612   -1.832   39.809
O   25.253   -2.889   38.605
N   22.437   -6.567   45.203
C   21.738   -5.659   44.496
C   20.438   -6.144   44.344
C   20.380   -7.565   44.789
C   21.713   -7.727   45.387
C   19.410   -5.251   43.633
C   19.207   -8.517   44.595
O   18.195   -8.070   44.114
C   19.284   -9.971   44.855
N   24.654   -8.222   46.472
C   23.597   -9.056   46.580
C   23.895   -10.314   47.416
C   25.450   -10.285   47.508
C   25.720   -8.834   47.087
C   23.090   -10.381   48.777
C   26.244   -11.208   46.578
C   26.952   -12.395   47.133
N   26.501   -6.177   46.363
C   27.355   -6.959   46.958
C   28.605   -6.336   47.215
C   28.487   -5.142   46.469
C   27.183   -5.160   45.919
C   29.755   -6.893   47.887
C   29.118   -3.829   46.065
O   30.221   -3.317   46.347
C   28.055   -3.137   45.054
C   27.864   -1.759   45.706
O   28.297   -0.669   45.373
O   26.987   -1.884   46.762
C   26.245   -0.720   47.314
C   24.303   -2.771   37.461
C   23.671   -4.034   36.946
C   24.241   -5.032   36.217
C   25.680   -5.125   35.703
C   23.410   -6.228   35.767
C   22.464   -6.019   34.547
C   22.571   -7.038   33.455
C   23.204   -6.470   32.144
C   22.571   -6.978   30.833
C   24.745   -6.736   31.965
C   25.451   -5.532   31.280
C   26.002   -5.976   29.916
C   25.464   -5.170   28.738
C   26.422   -3.964   28.714
C   25.387   -6.019   27.437
C   24.665   -5.295   26.283
C   23.137   -5.643   26.279
C   22.297   -4.331   26.474
C   21.537   -4.150   27.790
C   21.571   -4.033   25.153
H   21.570   -3.698   43.622
H   21.757   -9.764   46.149
H   27.681   -8.878   47.806
H   26.032   -1.766   43.878
H   23.538   -2.347   42.657
H   24.135   -0.780   44.979
H   22.957   -1.924   45.460
H   22.500   -0.963   44.105
H   25.521   -3.977   41.843
H   27.120   -3.154   42.229
H   26.817   -1.912   40.399
H   25.531   -1.128   41.383
H   19.290   -5.543   42.589
H   19.624   -4.183   43.603
H   18.507   -5.285   44.242
H   19.156   -10.194   45.914
H   20.224   -10.320   44.429
H   18.376   -10.402   44.434
H   23.574   -11.088   46.720
H   25.805   -10.527   48.509
H   23.628   -9.848   49.562
H   22.918   -11.436   48.988
H   22.172   -9.793   48.741
H   27.026   -10.606   46.115
H   25.549   -11.618   45.844
H   28.000   -12.412   46.834
H   26.524   -13.322   46.751
H   26.948   -12.221   48.209
H   29.471   -7.589   48.676
H   30.317   -6.046   48.281
H   30.335   -7.378   47.102
H   28.601   -3.000   44.120
H   25.358   -0.439   46.746
H   26.924   0.070   47.636
H   25.849   -1.042   48.277
H   24.944   -2.469   36.632
H   23.463   -2.088   37.592
H   22.600   -4.142   37.113
H   25.802   -5.610   34.735
H   26.345   -5.661   36.381
H   25.995   -4.098   35.518
H   22.734   -6.528   36.569
H   24.086   -7.078   35.678
H   22.385   -4.987   34.205
H   21.509   -6.293   34.995
H   21.546   -7.382   33.319
H   23.129   -7.891   33.843
H   23.033   -5.394   32.159
H   23.186   -7.599   30.183
H   22.268   -6.086   30.285
H   21.670   -7.556   31.038
H   24.822   -7.646   31.369
H   25.195   -7.061   32.903
H   26.210   -5.302   32.029
H   24.947   -4.565   31.266
H   25.676   -7.002   29.746
H   27.086   -5.863   29.961
H   24.468   -4.803   28.984
H   27.144   -4.220   27.939
H   26.959   -3.712   29.629
H   25.759   -3.138   28.458
H   24.908   -6.969   27.673
H   26.351   -6.461   27.184
H   25.116   -5.598   25.338
H   24.728   -4.207   26.324
H   22.863   -6.280   27.119
H   23.002   -6.132   25.314
H   22.874   -3.411   26.564
H   20.483   -4.251   27.532
H   21.796   -3.212   28.281
H   21.724   -4.974   28.478
H   20.604   -3.592   25.397
H   21.352   -4.882   24.504
H   22.072   -3.199   24.662

