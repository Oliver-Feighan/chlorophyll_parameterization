%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1_chromophore_26 ZINDO

0 1
Mg   -9.126   18.730   43.070
C   -5.623   18.445   42.867
C   -8.711   22.083   42.246
C   -12.431   19.033   42.793
C   -9.303   15.266   42.957
N   -7.320   20.025   42.453
C   -5.967   19.766   42.554
C   -5.070   20.986   42.549
C   -6.066   22.157   42.373
C   -7.448   21.426   42.335
C   -5.986   23.258   43.507
C   -3.936   21.031   41.430
C   -4.226   20.351   40.082
C   -4.566   21.252   38.858
O   -4.983   22.394   38.902
O   -4.154   20.569   37.738
N   -10.362   20.306   42.605
C   -10.015   21.604   42.359
C   -11.224   22.428   42.220
C   -12.336   21.546   42.382
C   -11.711   20.205   42.598
C   -11.197   23.932   41.975
C   -13.794   22.107   42.188
O   -13.918   23.281   41.868
C   -15.046   21.141   42.316
N   -10.602   17.364   42.777
C   -11.949   17.711   42.852
C   -12.887   16.376   42.871
C   -11.752   15.220   42.763
C   -10.485   16.009   42.845
C   -13.770   16.281   44.150
C   -11.807   14.336   41.444
C   -11.819   12.805   41.574
N   -7.762   17.118   43.062
C   -7.944   15.760   43.004
C   -6.617   15.110   42.965
C   -5.679   16.175   43.031
C   -6.456   17.367   43.038
C   -6.315   13.641   42.857
C   -4.231   16.453   42.988
O   -3.231   15.722   43.038
C   -4.164   17.970   43.083
C   -3.542   18.424   44.281
O   -4.154   18.455   45.380
O   -2.218   18.846   44.048
C   -1.496   19.491   45.140
C   -4.399   21.306   36.473
C   -5.725   20.955   35.967
C   -6.137   19.839   35.287
C   -5.182   18.625   34.982
C   -7.544   19.696   34.631
C   -8.371   18.342   34.658
C   -9.373   18.281   33.396
C   -10.574   19.305   33.518
C   -12.010   18.792   33.242
C   -10.234   20.449   32.520
C   -10.952   21.722   32.660
C   -10.067   23.010   32.543
C   -10.386   24.031   31.351
C   -9.902   25.492   31.781
C   -9.853   23.521   29.945
C   -10.927   23.089   28.928
C   -10.649   23.747   27.590
C   -11.247   22.823   26.473
C   -10.677   23.281   25.195
C   -12.813   23.069   26.424
H   -8.639   23.169   42.161
H   -13.517   19.116   42.866
H   -9.445   14.184   42.983
H   -4.615   20.947   43.538
H   -5.923   22.509   41.352
H   -5.582   24.180   43.089
H   -5.326   22.850   44.273
H   -6.972   23.481   43.916
H   -3.141   20.464   41.916
H   -3.607   22.062   41.298
H   -5.037   19.629   40.183
H   -3.369   19.729   39.827
H   -11.415   24.232   40.950
H   -10.312   24.386   42.421
H   -11.944   24.436   42.588
H   -14.909   20.569   43.234
H   -15.050   20.497   41.437
H   -16.006   21.639   42.180
H   -13.597   16.501   42.054
H   -11.833   14.613   43.665
H   -13.233   15.665   44.871
H   -14.782   15.934   43.941
H   -13.987   17.243   44.614
H   -10.980   14.667   40.815
H   -12.761   14.648   41.019
H   -11.511   12.313   40.651
H   -12.866   12.602   41.798
H   -11.126   12.603   42.391
H   -7.190   13.229   42.355
H   -6.334   13.325   43.900
H   -5.341   13.395   42.433
H   -3.567   18.212   42.204
H   -1.276   18.888   46.021
H   -2.039   20.376   45.473
H   -0.613   19.949   44.694
H   -3.676   21.036   35.703
H   -4.307   22.369   36.695
H   -6.495   21.722   36.058
H   -5.377   17.629   35.380
H   -4.238   18.924   35.438
H   -5.161   18.589   33.893
H   -7.388   20.052   33.613
H   -8.198   20.459   35.052
H   -8.936   18.469   35.581
H   -7.766   17.435   34.655
H   -9.842   17.300   33.317
H   -8.822   18.333   32.457
H   -10.601   19.869   34.451
H   -12.414   19.139   32.291
H   -12.603   19.111   34.099
H   -12.128   17.728   33.037
H   -10.602   19.957   31.619
H   -9.147   20.535   32.495
H   -11.314   21.746   33.688
H   -11.710   21.831   31.884
H   -9.041   22.649   32.464
H   -10.156   23.530   33.497
H   -11.451   24.201   31.196
H   -9.277   25.944   31.010
H   -9.433   25.501   32.765
H   -10.762   26.111   32.037
H   -9.307   22.588   30.088
H   -9.114   24.234   29.577
H   -11.878   23.463   29.306
H   -11.043   22.006   28.975
H   -9.564   23.803   27.501
H   -11.018   24.755   27.407
H   -11.044   21.759   26.592
H   -10.832   24.356   25.097
H   -11.183   22.865   24.324
H   -9.607   23.080   25.141
H   -13.052   23.493   27.399
H   -13.380   22.146   26.302
H   -13.140   23.826   25.711

