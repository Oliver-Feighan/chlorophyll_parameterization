%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1151_chromophore_18 TDDFT with cam-b3lyp functional

0 1
Mg   34.824   50.154   25.511
C   34.824   48.403   28.594
C   33.344   52.849   27.120
C   34.550   51.572   22.525
C   35.822   47.160   23.973
N   34.463   50.651   27.769
C   34.458   49.815   28.771
C   33.966   50.455   30.096
C   33.436   51.878   29.515
C   33.789   51.828   28.027
C   31.931   52.251   29.912
C   35.097   50.478   31.118
C   34.734   49.888   32.489
C   33.255   49.928   32.862
O   32.503   48.955   32.930
O   32.841   51.223   32.982
N   34.168   52.013   24.955
C   33.555   52.951   25.729
C   33.180   54.088   24.933
C   33.560   53.760   23.575
C   34.142   52.421   23.644
C   32.187   55.171   25.391
C   33.490   54.580   22.315
O   33.863   54.138   21.234
C   33.016   56.053   22.389
N   35.147   49.387   23.519
C   34.965   50.243   22.454
C   35.222   49.502   21.068
C   35.764   48.111   21.596
C   35.581   48.244   23.106
C   33.943   49.397   20.217
C   37.206   47.726   21.144
C   38.468   48.350   21.901
N   35.249   48.178   26.094
C   35.688   47.096   25.384
C   35.928   45.982   26.243
C   35.614   46.443   27.572
C   35.191   47.777   27.376
C   36.468   44.646   25.912
C   35.584   46.061   28.943
O   36.063   45.105   29.456
C   34.961   47.402   29.728
C   33.671   47.018   30.368
O   32.559   47.498   30.132
O   33.851   46.140   31.473
C   32.692   45.505   32.036
C   31.405   51.381   33.352
C   31.183   52.740   33.854
C   30.977   53.208   35.144
C   30.969   52.283   36.386
C   30.682   54.639   35.496
C   31.767   55.627   35.141
C   31.274   56.989   34.582
C   31.476   58.200   35.611
C   32.920   58.739   35.752
C   30.513   59.336   35.428
C   29.448   59.465   36.497
C   29.835   60.178   37.787
C   29.282   61.721   37.731
C   30.306   62.642   38.383
C   27.820   61.871   38.135
C   26.981   62.528   37.024
C   25.547   62.931   37.581
C   25.217   64.404   37.643
C   25.209   65.004   39.022
C   23.984   64.802   36.753
H   32.725   53.658   27.513
H   34.318   51.938   21.523
H   36.142   46.287   23.400
H   33.088   49.960   30.511
H   33.996   52.619   30.085
H   32.008   52.996   30.704
H   31.498   51.288   30.182
H   31.346   52.642   29.079
H   35.380   51.520   31.266
H   36.035   50.068   30.745
H   35.234   50.525   33.219
H   35.136   48.882   32.369
H   32.725   55.995   25.861
H   31.403   54.763   26.029
H   31.459   55.573   24.687
H   33.619   56.521   23.168
H   31.969   56.047   22.692
H   33.211   56.478   21.405
H   35.980   50.196   20.705
H   35.083   47.319   21.285
H   33.079   49.883   20.671
H   33.604   48.390   19.974
H   34.078   49.951   19.289
H   37.263   48.190   20.160
H   37.395   46.653   21.109
H   38.131   48.880   22.791
H   39.081   48.988   21.264
H   39.110   47.549   22.267
H   37.226   44.757   25.136
H   35.666   43.985   25.585
H   36.945   44.312   26.833
H   35.622   47.717   30.535
H   32.401   44.683   31.382
H   31.777   46.070   32.214
H   33.127   45.075   32.938
H   31.207   50.644   34.131
H   30.731   51.226   32.510
H   30.946   53.493   33.101
H   31.838   52.440   37.024
H   30.911   51.250   36.042
H   30.078   52.625   36.911
H   30.497   54.782   36.561
H   29.763   54.944   34.995
H   32.474   55.193   34.434
H   32.284   55.704   36.098
H   30.211   57.078   34.360
H   31.825   57.238   33.675
H   31.306   57.682   36.555
H   33.639   57.925   35.654
H   33.230   59.413   36.550
H   32.929   59.237   34.783
H   30.134   59.244   34.410
H   31.066   60.271   35.517
H   29.053   58.506   36.834
H   28.500   59.835   36.107
H   30.911   60.106   37.939
H   29.483   59.812   38.752
H   29.326   61.899   36.657
H   29.852   63.242   39.171
H   30.661   63.253   37.553
H   31.098   62.096   38.896
H   27.735   62.496   39.025
H   27.419   60.864   38.252
H   26.832   61.878   36.162
H   27.457   63.429   36.636
H   25.356   62.536   38.579
H   24.797   62.396   37.000
H   25.985   64.965   37.111
H   26.145   65.561   39.067
H   25.240   64.273   39.830
H   24.501   65.822   39.155
H   24.388   64.968   35.754
H   23.516   65.670   37.217
H   23.289   63.999   36.508

