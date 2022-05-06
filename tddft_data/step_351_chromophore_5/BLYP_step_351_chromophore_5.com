%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_351_chromophore_5 TDDFT with blyp functional

0 1
Mg   23.887   -6.763   45.575
C   26.199   -4.435   44.283
C   21.441   -5.134   43.706
C   21.882   -9.344   46.252
C   26.635   -8.459   46.911
N   23.905   -5.127   43.942
C   24.949   -4.336   43.658
C   24.446   -3.210   42.674
C   22.873   -3.283   42.725
C   22.700   -4.572   43.506
C   22.241   -2.061   43.418
C   25.079   -3.503   41.268
C   25.563   -2.379   40.427
C   25.286   -2.465   38.880
O   24.703   -1.564   38.262
O   25.695   -3.618   38.299
N   21.895   -7.176   45.121
C   21.060   -6.367   44.349
C   19.729   -6.944   44.451
C   19.809   -8.142   45.188
C   21.218   -8.262   45.596
C   18.580   -6.371   43.691
C   18.629   -8.994   45.482
O   17.501   -8.689   45.032
C   18.759   -10.100   46.314
N   24.165   -8.713   46.462
C   23.203   -9.554   46.682
C   23.787   -10.767   47.408
C   25.208   -10.299   47.918
C   25.363   -9.060   46.987
C   22.921   -11.435   48.437
C   26.309   -11.419   47.763
C   27.048   -11.686   49.122
N   26.020   -6.511   45.637
C   26.967   -7.250   46.291
C   28.223   -6.735   46.037
C   28.027   -5.571   45.283
C   26.621   -5.521   45.062
C   29.540   -7.208   46.581
C   28.636   -4.374   44.660
O   29.799   -4.057   44.543
C   27.480   -3.599   43.958
C   27.466   -2.246   44.456
O   27.716   -1.276   43.749
O   27.012   -2.215   45.704
C   26.733   -0.893   46.278
C   25.401   -3.773   36.874
C   25.130   -5.251   36.720
C   25.633   -6.138   35.887
C   27.067   -6.002   35.335
C   25.085   -7.617   35.687
C   23.647   -7.908   35.106
C   23.575   -8.545   33.767
C   23.118   -7.473   32.714
C   22.349   -8.208   31.607
C   24.394   -6.735   32.200
C   24.285   -5.158   32.217
C   25.455   -4.489   31.373
C   25.012   -3.741   30.049
C   25.018   -2.241   30.326
C   26.002   -4.148   28.902
C   25.304   -4.956   27.785
C   24.564   -4.005   26.811
C   23.137   -4.435   26.448
C   23.034   -4.938   24.930
C   22.141   -3.304   26.730
H   20.622   -4.649   43.171
H   21.314   -10.271   46.362
H   27.510   -8.918   47.377
H   24.800   -2.254   43.060
H   22.501   -3.305   41.700
H   21.938   -2.479   44.378
H   21.298   -1.927   42.887
H   22.840   -1.174   43.627
H   24.275   -3.920   40.661
H   25.770   -4.345   41.300
H   26.627   -2.242   40.618
H   25.055   -1.489   40.798
H   18.671   -6.633   42.637
H   18.706   -5.288   43.688
H   17.610   -6.564   44.149
H   19.429   -9.737   47.094
H   19.280   -10.892   45.777
H   17.788   -10.421   46.690
H   24.027   -11.498   46.635
H   25.077   -10.026   48.965
H   21.900   -11.062   48.516
H   23.365   -11.197   49.404
H   22.832   -12.496   48.204
H   26.956   -11.195   46.915
H   25.802   -12.343   47.486
H   27.005   -12.699   49.522
H   26.766   -11.017   49.935
H   28.132   -11.622   49.028
H   29.953   -8.073   46.061
H   29.301   -7.500   47.604
H   30.300   -6.428   46.627
H   27.596   -3.625   42.875
H   26.174   -0.904   47.214
H   26.096   -0.357   45.576
H   27.686   -0.368   46.351
H   26.272   -3.490   36.283
H   24.586   -3.160   36.489
H   24.133   -5.517   37.072
H   26.959   -5.488   34.380
H   27.652   -6.921   35.298
H   27.765   -5.283   35.764
H   25.143   -8.037   36.692
H   25.875   -8.106   35.118
H   23.039   -7.004   35.062
H   23.089   -8.566   35.771
H   22.862   -9.366   33.833
H   24.504   -9.018   33.450
H   22.354   -6.784   33.075
H   21.386   -7.756   31.371
H   22.345   -9.292   31.722
H   22.935   -8.056   30.701
H   24.656   -7.071   31.197
H   25.235   -7.052   32.816
H   24.532   -4.822   33.224
H   23.314   -4.763   31.919
H   26.234   -5.251   31.402
H   25.937   -3.756   32.019
H   23.989   -3.868   29.695
H   24.394   -2.138   31.214
H   24.727   -1.718   29.415
H   25.998   -1.906   30.664
H   26.840   -4.789   29.176
H   26.423   -3.311   28.346
H   24.615   -5.657   28.257
H   25.933   -5.620   27.193
H   25.131   -4.156   25.893
H   24.687   -2.956   27.082
H   22.807   -5.211   27.139
H   22.350   -5.783   24.848
H   23.971   -5.086   24.392
H   22.476   -4.166   24.400
H   21.334   -3.626   27.388
H   21.698   -2.918   25.811
H   22.451   -2.387   27.231

