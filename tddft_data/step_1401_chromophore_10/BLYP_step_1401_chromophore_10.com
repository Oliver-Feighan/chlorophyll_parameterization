%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1401_chromophore_10 TDDFT with blyp functional

0 1
Mg   40.950   7.827   29.508
C   42.685   9.282   32.201
C   38.720   6.566   31.700
C   39.667   6.209   26.980
C   43.620   8.842   27.494
N   40.709   7.998   31.711
C   41.549   8.609   32.596
C   41.050   8.498   33.998
C   39.993   7.363   33.873
C   39.743   7.369   32.286
C   40.359   5.918   34.333
C   40.457   9.849   34.541
C   40.002   9.927   36.037
C   40.800   10.506   37.155
O   41.884   11.097   37.184
O   40.012   10.320   38.308
N   39.459   6.561   29.382
C   38.546   6.269   30.365
C   37.383   5.578   29.905
C   37.582   5.497   28.478
C   39.002   6.096   28.216
C   36.259   4.910   30.728
C   36.582   5.004   27.352
O   36.883   4.935   26.218
C   35.210   4.620   27.774
N   41.639   7.347   27.585
C   40.765   6.865   26.687
C   41.309   6.929   25.309
C   42.390   8.071   25.453
C   42.587   8.078   26.907
C   41.867   5.553   24.795
C   41.897   9.390   24.692
C   40.838   10.255   25.372
N   42.731   8.894   29.697
C   43.771   9.222   28.797
C   44.830   9.992   29.426
C   44.394   10.126   30.772
C   43.207   9.383   30.892
C   46.100   10.519   28.800
C   44.768   10.626   32.031
O   45.852   11.191   32.369
C   43.664   10.205   33.042
C   44.381   9.410   34.076
O   45.186   8.536   33.936
O   44.013   9.872   35.319
C   44.556   9.118   36.446
C   40.648   10.723   39.641
C   40.210   9.680   40.629
C   40.396   9.732   41.954
C   40.978   10.840   42.854
C   39.862   8.593   42.849
C   38.336   8.771   43.151
C   38.090   8.801   44.677
C   37.941   10.242   45.236
C   38.469   10.322   46.692
C   36.447   10.769   45.178
C   36.128   12.059   44.358
C   34.727   12.047   43.775
C   33.742   13.055   44.428
C   33.422   12.701   45.888
C   32.444   13.283   43.568
C   32.285   14.676   42.976
C   32.326   14.553   41.443
C   30.906   14.531   40.821
C   30.748   13.302   39.861
C   30.384   15.865   40.163
H   37.965   6.198   32.398
H   39.126   5.769   26.140
H   44.421   9.076   26.790
H   41.916   8.155   34.565
H   39.011   7.594   34.286
H   40.265   5.280   33.454
H   39.696   5.573   35.127
H   41.397   5.929   34.665
H   39.506   10.041   34.045
H   41.035   10.761   34.388
H   39.690   8.944   36.389
H   39.128   10.562   36.187
H   35.344   5.455   30.495
H   36.443   4.841   31.800
H   36.107   3.886   30.386
H   35.140   3.571   28.059
H   34.644   4.635   26.842
H   34.685   5.333   28.409
H   40.502   7.332   24.697
H   43.304   7.886   24.890
H   42.955   5.569   24.734
H   41.280   5.408   23.888
H   41.522   4.835   25.540
H   41.399   9.027   23.793
H   42.749   10.033   24.470
H   40.199   9.704   26.062
H   40.128   10.497   24.582
H   41.276   11.145   25.825
H   46.833   9.722   28.926
H   46.559   11.243   29.474
H   45.878   10.794   27.769
H   43.237   11.084   33.526
H   43.991   8.196   36.311
H   44.471   9.600   37.420
H   45.630   8.930   36.429
H   40.251   11.644   40.069
H   41.729   10.617   39.727
H   39.806   8.778   40.168
H   41.440   10.426   43.750
H   40.294   11.670   43.032
H   41.899   11.260   42.449
H   40.465   8.554   43.756
H   40.109   7.658   42.347
H   37.677   7.973   42.807
H   37.994   9.702   42.699
H   38.847   8.236   45.221
H   37.158   8.247   44.791
H   38.560   10.890   44.615
H   37.904   9.483   47.098
H   38.134   11.263   47.127
H   39.553   10.222   46.635
H   36.092   10.975   46.188
H   35.896   9.926   44.760
H   36.899   12.334   43.639
H   36.173   12.858   45.099
H   34.276   11.104   44.085
H   34.690   12.124   42.688
H   34.364   13.948   44.480
H   32.938   13.554   46.363
H   34.357   12.468   46.398
H   32.862   11.768   45.956
H   31.578   13.132   44.212
H   32.288   12.494   42.832
H   33.122   15.364   43.096
H   31.395   15.187   43.343
H   32.966   13.686   41.279
H   32.882   15.369   40.981
H   30.190   14.219   41.581
H   31.355   12.439   40.135
H   30.983   13.544   38.825
H   29.720   12.943   39.931
H   30.778   16.178   39.196
H   30.556   16.608   40.941
H   29.297   15.927   40.205

