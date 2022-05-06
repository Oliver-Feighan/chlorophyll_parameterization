%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1_chromophore_1 TDDFT with blyp functional

0 1
Mg   -1.968   16.991   27.078
C   -2.185   14.773   29.879
C   -3.223   19.433   29.104
C   -2.213   18.841   24.514
C   -1.822   14.167   25.041
N   -2.585   16.982   29.204
C   -2.562   16.083   30.183
C   -2.904   16.682   31.580
C   -3.741   17.955   31.110
C   -3.123   18.159   29.738
C   -5.231   17.676   30.941
C   -1.623   17.101   32.377
C   -1.844   17.388   33.865
C   -0.800   17.024   34.811
O   0.398   16.855   34.599
O   -1.380   16.867   36.028
N   -2.153   18.952   26.926
C   -2.671   19.791   27.855
C   -2.588   21.147   27.271
C   -2.155   21.006   25.853
C   -2.182   19.574   25.698
C   -2.855   22.397   28.052
C   -1.787   22.004   24.801
O   -1.665   21.681   23.570
C   -1.658   23.405   25.241
N   -2.233   16.578   25.049
C   -2.247   17.560   24.141
C   -2.164   17.015   22.746
C   -1.844   15.458   22.898
C   -1.882   15.359   24.443
C   -3.419   17.196   21.881
C   -0.595   14.921   22.155
C   0.658   15.501   22.772
N   -1.818   14.847   27.381
C   -1.795   13.855   26.390
C   -1.795   12.553   27.082
C   -1.835   12.880   28.436
C   -1.885   14.296   28.580
C   -1.819   11.276   26.455
C   -1.872   12.362   29.794
O   -1.760   11.244   30.299
C   -2.223   13.563   30.779
C   -1.251   13.509   31.926
O   -0.037   13.736   31.747
O   -1.930   13.435   33.081
C   -1.025   13.408   34.261
C   -0.516   16.468   37.138
C   -0.531   17.387   38.262
C   -0.454   17.127   39.590
C   -0.378   15.838   40.321
C   -0.723   18.255   40.576
C   0.451   18.789   41.502
C   1.210   20.031   40.870
C   1.469   21.168   41.942
C   0.066   21.921   42.199
C   2.756   22.039   41.670
C   3.243   22.867   42.872
C   4.766   22.908   42.988
C   5.267   23.085   44.478
C   6.800   23.421   44.531
C   4.850   21.941   45.489
C   3.950   22.297   46.696
C   2.527   21.696   46.703
C   1.426   22.715   47.155
C   0.033   22.091   47.443
C   1.232   23.982   46.288
H   -3.827   20.211   29.575
H   -2.186   19.444   23.604
H   -1.692   13.244   24.473
H   -3.522   16.042   32.210
H   -3.553   18.822   31.743
H   -5.452   16.614   31.047
H   -5.479   18.018   29.936
H   -5.820   18.292   31.621
H   -1.299   18.087   32.044
H   -0.838   16.352   32.267
H   -2.812   17.017   34.203
H   -1.906   18.468   33.999
H   -3.312   22.177   29.016
H   -3.619   22.954   27.510
H   -1.961   23.011   28.162
H   -2.603   23.735   25.673
H   -1.303   23.961   24.374
H   -0.920   23.457   26.042
H   -1.360   17.492   22.187
H   -2.660   14.835   22.531
H   -3.235   17.731   20.949
H   -4.172   17.705   22.482
H   -3.935   16.271   21.625
H   -0.612   15.232   21.111
H   -0.516   13.834   22.176
H   0.746   16.469   22.280
H   1.519   14.870   22.549
H   0.527   15.688   23.838
H   -1.101   10.658   26.994
H   -1.564   11.349   25.398
H   -2.859   10.992   26.615
H   -3.251   13.410   31.106
H   -1.270   12.513   34.833
H   -1.135   14.372   34.758
H   0.032   13.237   34.054
H   0.552   16.428   36.922
H   -0.907   15.476   37.361
H   -1.044   18.319   38.023
H   -0.543   14.989   39.658
H   -1.182   15.731   41.050
H   0.581   15.668   40.812
H   -1.499   17.903   41.256
H   -1.227   19.095   40.097
H   1.292   18.096   41.508
H   0.091   19.058   42.494
H   0.620   20.446   40.053
H   2.115   19.576   40.468
H   1.778   20.639   42.843
H   0.034   22.900   41.721
H   -0.263   21.889   43.237
H   -0.693   21.476   41.556
H   2.393   22.719   40.900
H   3.556   21.429   41.250
H   2.826   22.458   43.793
H   2.891   23.896   42.798
H   5.207   23.572   42.244
H   5.175   21.964   42.627
H   4.875   24.075   44.715
H   7.330   22.497   44.300
H   7.166   23.617   45.539
H   7.126   24.262   43.920
H   5.826   21.689   45.904
H   4.491   21.050   44.973
H   3.897   23.383   46.781
H   4.392   21.943   47.627
H   2.514   20.877   47.422
H   2.203   21.325   45.731
H   1.782   22.975   48.151
H   -0.337   22.156   48.466
H   0.258   21.029   47.348
H   -0.654   22.475   46.688
H   1.826   23.915   45.376
H   1.813   24.727   46.832
H   0.234   24.277   45.960
