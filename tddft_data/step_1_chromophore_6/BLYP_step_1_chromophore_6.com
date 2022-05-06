%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1_chromophore_6 TDDFT with blyp functional

0 1
Mg   17.054   -2.288   27.750
C   16.548   -0.354   30.726
C   18.852   -4.529   29.657
C   18.013   -3.670   24.949
C   15.392   0.222   26.033
N   17.886   -2.277   29.962
C   17.412   -1.437   30.933
C   17.782   -2.001   32.331
C   18.585   -3.327   31.996
C   18.484   -3.424   30.430
C   20.102   -3.411   32.453
C   16.578   -2.286   33.268
C   16.684   -1.850   34.686
C   18.053   -1.523   35.396
O   18.778   -0.591   35.104
O   18.548   -2.518   36.236
N   18.305   -3.918   27.310
C   18.859   -4.819   28.219
C   19.641   -5.825   27.475
C   19.293   -5.632   26.083
C   18.516   -4.371   26.058
C   20.531   -6.871   28.075
C   19.639   -6.463   24.825
O   19.045   -6.384   23.729
C   20.630   -7.546   24.977
N   16.791   -1.754   25.844
C   17.373   -2.499   24.842
C   16.845   -2.070   23.445
C   15.703   -1.041   23.862
C   15.953   -0.843   25.347
C   17.880   -1.479   22.403
C   14.265   -1.461   23.602
C   13.502   -0.619   22.558
N   15.981   -0.538   28.255
C   15.386   0.394   27.438
C   14.878   1.514   28.241
C   15.273   1.207   29.531
C   15.938   -0.039   29.497
C   14.172   2.751   27.793
C   15.189   1.699   30.859
O   14.574   2.682   31.362
C   15.981   0.758   31.664
C   17.012   1.503   32.468
O   17.975   2.031   32.027
O   16.720   1.423   33.766
C   17.368   2.297   34.684
C   19.982   -2.452   36.552
C   20.408   -3.764   37.247
C   20.356   -4.021   38.546
C   19.854   -3.056   39.583
C   20.837   -5.333   39.057
C   22.258   -5.655   39.002
C   22.740   -6.699   40.128
C   23.734   -7.691   39.446
C   22.955   -8.755   38.715
C   24.769   -8.272   40.501
C   26.123   -7.521   40.532
C   26.806   -7.793   41.936
C   27.491   -9.180   41.978
C   26.842   -9.995   43.104
C   29.044   -9.062   42.251
C   29.902   -9.093   40.942
C   31.024   -10.168   41.003
C   31.344   -10.751   39.606
C   32.091   -9.703   38.807
C   32.157   -12.063   39.749
H   19.445   -5.281   30.182
H   18.215   -4.046   23.943
H   14.940   1.043   25.474
H   18.422   -1.249   32.794
H   18.070   -4.189   32.419
H   20.439   -2.604   33.103
H   20.706   -3.311   31.550
H   20.311   -4.382   32.903
H   16.349   -3.349   33.352
H   15.660   -1.804   32.933
H   16.309   -2.637   35.341
H   16.049   -0.972   34.801
H   21.524   -6.798   27.632
H   20.096   -7.833   27.804
H   20.474   -6.870   29.164
H   21.579   -7.085   25.251
H   20.755   -7.963   23.978
H   20.262   -8.353   25.610
H   16.370   -2.960   23.031
H   15.888   -0.125   23.301
H   18.880   -1.465   22.835
H   17.571   -0.518   21.990
H   17.864   -2.133   21.531
H   13.635   -1.616   24.478
H   14.307   -2.459   23.167
H   12.479   -0.335   22.804
H   13.298   -1.281   21.716
H   13.992   0.304   22.251
H   13.755   2.656   26.790
H   14.889   3.560   27.654
H   13.497   3.030   28.602
H   15.272   0.271   32.333
H   16.919   3.280   34.535
H   18.417   2.432   34.420
H   17.174   2.000   35.715
H   20.182   -1.599   37.201
H   20.542   -2.446   35.617
H   20.690   -4.616   36.628
H   19.138   -2.446   39.033
H   20.664   -2.451   39.990
H   19.457   -3.682   40.382
H   20.285   -6.003   38.398
H   20.431   -5.496   40.055
H   22.825   -4.727   38.932
H   22.485   -6.036   38.007
H   21.940   -7.281   40.587
H   23.253   -6.066   40.853
H   24.429   -7.174   38.785
H   23.377   -8.933   37.726
H   21.938   -8.390   38.577
H   22.868   -9.662   39.313
H   24.871   -9.356   40.539
H   24.367   -7.952   41.462
H   25.835   -6.482   40.376
H   26.762   -7.954   39.763
H   26.064   -7.770   42.733
H   27.439   -6.930   42.144
H   27.284   -9.649   41.016
H   27.200   -9.702   44.091
H   27.061   -11.049   42.931
H   25.773   -9.788   43.068
H   29.359   -9.968   42.769
H   29.234   -8.137   42.796
H   30.419   -8.134   40.972
H   29.280   -9.128   40.047
H   30.802   -11.040   41.618
H   31.870   -9.865   41.621
H   30.338   -10.853   39.200
H   31.684   -9.983   37.835
H   33.179   -9.737   38.747
H   31.830   -8.679   39.075
H   33.195   -11.825   39.514
H   31.775   -12.699   38.950
H   32.036   -12.580   40.701

