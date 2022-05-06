%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1451_chromophore_10 TDDFT with blyp functional

0 1
Mg   40.834   7.987   29.614
C   42.575   9.419   32.323
C   38.618   6.458   31.841
C   39.231   6.484   26.929
C   43.138   9.465   27.441
N   40.572   8.010   31.761
C   41.388   8.714   32.635
C   40.818   8.682   34.081
C   39.732   7.538   34.004
C   39.607   7.295   32.423
C   40.142   6.309   34.886
C   40.118   9.993   34.456
C   39.619   10.048   35.897
C   40.264   11.067   36.842
O   40.317   12.327   36.771
O   40.912   10.417   37.889
N   39.160   6.724   29.426
C   38.357   6.235   30.442
C   37.220   5.546   29.932
C   37.324   5.562   28.533
C   38.616   6.238   28.208
C   36.234   4.824   30.827
C   36.190   5.101   27.529
O   36.345   5.102   26.316
C   34.929   4.836   28.047
N   41.192   7.947   27.501
C   40.339   7.259   26.659
C   40.820   7.453   25.220
C   41.877   8.617   25.318
C   42.087   8.718   26.857
C   41.309   6.076   24.647
C   41.379   9.959   24.686
C   40.158   10.651   25.390
N   42.605   9.067   29.800
C   43.447   9.590   28.836
C   44.603   10.137   29.468
C   44.250   10.221   30.837
C   43.060   9.496   31.010
C   45.846   10.718   28.779
C   44.694   10.619   32.162
O   45.766   11.128   32.540
C   43.607   10.119   33.200
C   44.222   9.332   34.319
O   45.197   8.680   34.209
O   43.612   9.585   35.524
C   44.045   8.789   36.666
C   41.531   11.307   38.890
C   41.142   10.889   40.355
C   41.755   11.345   41.532
C   42.871   12.396   41.713
C   41.272   10.780   42.924
C   40.158   11.494   43.662
C   38.794   11.153   43.050
C   37.739   10.756   44.166
C   37.969   9.357   44.713
C   36.309   10.863   43.643
C   35.712   12.202   43.673
C   34.444   12.256   44.524
C   34.591   13.138   45.840
C   35.214   12.236   46.799
C   33.261   13.839   46.278
C   33.150   15.151   45.526
C   31.763   15.466   45.015
C   31.765   15.595   43.464
C   31.760   17.118   42.975
C   30.470   14.980   42.837
H   37.898   6.041   32.547
H   38.706   6.231   26.005
H   43.782   9.826   26.637
H   41.523   8.362   34.849
H   38.792   7.892   34.426
H   39.511   6.355   35.774
H   41.165   6.329   35.260
H   40.021   5.372   34.343
H   39.314   10.127   33.732
H   40.755   10.861   34.281
H   39.635   9.042   36.314
H   38.554   10.281   35.875
H   36.599   4.603   31.830
H   35.853   3.845   30.538
H   35.381   5.497   30.911
H   34.931   3.946   28.676
H   34.276   4.623   27.200
H   34.691   5.656   28.724
H   39.994   7.804   24.602
H   42.775   8.251   24.820
H   42.342   5.907   24.953
H   41.159   5.983   23.571
H   40.760   5.187   24.958
H   41.101   9.782   23.647
H   42.279   10.558   24.548
H   39.269   10.036   25.250
H   39.965   11.684   25.102
H   40.452   10.822   26.425
H   46.670   10.025   28.950
H   46.098   11.648   29.289
H   45.680   10.947   27.726
H   43.004   10.873   33.707
H   44.986   8.242   36.605
H   43.341   8.044   37.037
H   44.239   9.522   37.449
H   41.218   12.349   38.817
H   42.608   11.256   38.731
H   40.408   10.105   40.541
H   42.336   13.337   41.842
H   43.533   12.400   40.847
H   43.420   12.197   42.634
H   42.163   10.673   43.543
H   40.868   9.784   42.748
H   40.279   12.561   43.476
H   40.237   11.251   44.722
H   38.756   10.256   42.432
H   38.524   11.965   42.376
H   37.904   11.439   44.999
H   37.426   8.640   44.098
H   37.571   9.161   45.709
H   39.042   9.194   44.817
H   35.665   10.103   44.085
H   36.239   10.653   42.575
H   35.462   12.509   42.657
H   36.439   12.876   44.127
H   34.053   11.252   44.691
H   33.676   12.746   43.925
H   35.323   13.941   45.749
H   36.141   12.677   47.165
H   35.478   11.243   46.435
H   34.487   11.991   47.573
H   33.420   14.085   47.328
H   32.383   13.214   46.114
H   33.959   15.145   44.795
H   33.462   15.844   46.308
H   31.638   16.462   45.439
H   31.011   14.723   45.279
H   32.632   15.044   43.099
H   30.809   17.648   43.036
H   32.185   17.182   41.973
H   32.343   17.736   43.657
H   29.688   15.716   42.648
H   30.083   14.199   43.492
H   30.836   14.451   41.958

