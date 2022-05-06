%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1801_chromophore_17 TDDFT with PBE1PBE functional

0 1
Mg   29.381   59.387   40.603
C   26.352   57.707   40.132
C   30.986   56.603   39.220
C   32.212   61.086   40.664
C   27.550   62.308   41.420
N   28.670   57.480   39.587
C   27.424   56.939   39.639
C   27.392   55.637   38.808
C   28.933   55.187   38.789
C   29.580   56.478   39.275
C   29.288   53.886   39.602
C   26.738   55.759   37.429
C   25.805   54.608   37.000
C   25.810   54.044   35.614
O   26.036   52.834   35.424
O   25.532   54.922   34.659
N   31.347   58.930   39.963
C   31.781   57.692   39.558
C   33.197   57.708   39.462
C   33.617   59.048   39.990
C   32.398   59.718   40.290
C   34.084   56.624   38.846
C   34.963   59.536   40.161
O   35.888   58.825   39.849
C   35.279   60.845   40.816
N   29.749   61.469   40.822
C   31.050   61.871   40.904
C   31.067   63.351   41.394
C   29.580   63.728   41.550
C   28.906   62.444   41.304
C   32.099   63.674   42.517
C   29.180   64.987   40.703
C   28.466   66.146   41.478
N   27.421   59.947   40.813
C   26.759   61.153   41.172
C   25.336   60.917   41.256
C   25.133   59.574   40.970
C   26.420   59.068   40.617
C   24.339   61.933   41.644
C   24.071   58.473   40.818
O   22.863   58.497   41.066
C   24.889   57.296   40.227
C   24.712   56.234   41.198
O   24.666   56.330   42.392
O   24.635   54.998   40.574
C   24.801   53.830   41.459
C   25.695   54.428   33.319
C   25.653   55.439   32.186
C   25.906   55.352   30.880
C   26.196   54.095   30.196
C   25.700   56.612   29.906
C   26.729   57.774   29.941
C   27.279   58.234   28.574
C   28.826   57.918   28.476
C   29.110   56.744   27.541
C   29.630   59.213   27.947
C   30.905   59.491   28.785
C   32.183   59.281   27.916
C   33.145   58.502   28.827
C   34.292   59.321   29.243
C   33.660   57.251   28.043
C   34.089   56.154   29.015
C   35.577   55.840   28.879
C   35.984   54.436   29.367
C   37.217   54.540   30.209
C   36.207   53.433   28.162
H   31.517   55.786   38.728
H   33.084   61.736   40.756
H   27.089   63.278   41.618
H   26.842   54.962   39.464
H   29.260   54.948   37.777
H   29.768   54.069   40.564
H   29.962   53.313   38.964
H   28.460   53.186   39.710
H   27.561   55.826   36.717
H   26.256   56.736   37.384
H   24.749   54.853   37.116
H   25.881   53.770   37.693
H   33.656   55.784   38.299
H   34.604   56.219   39.713
H   34.892   57.024   38.234
H   35.171   61.628   40.065
H   36.337   60.697   41.033
H   34.778   60.933   41.780
H   31.544   63.997   40.657
H   29.383   63.940   42.601
H   32.390   62.675   42.842
H   31.625   64.173   43.363
H   32.975   64.216   42.162
H   28.451   64.698   39.946
H   29.988   65.446   40.132
H   27.395   66.150   41.272
H   29.025   67.076   41.376
H   28.515   65.877   42.533
H   24.686   62.889   41.254
H   24.232   61.923   42.729
H   23.404   61.556   41.230
H   24.488   56.867   39.309
H   24.461   52.955   40.906
H   24.167   53.915   42.341
H   25.841   53.579   41.665
H   24.966   53.638   33.141
H   26.703   54.023   33.224
H   25.454   56.480   32.438
H   27.145   54.236   29.677
H   25.382   53.814   29.528
H   26.386   53.265   30.876
H   24.709   57.011   30.119
H   25.513   56.306   28.876
H   27.535   57.640   30.662
H   26.127   58.639   30.222
H   27.093   59.300   28.442
H   26.710   57.865   27.721
H   29.247   57.679   29.452
H   29.061   55.805   28.092
H   30.071   56.806   27.031
H   28.397   56.819   26.720
H   29.010   60.101   28.071
H   29.858   59.164   26.882
H   30.792   58.867   29.671
H   30.773   60.523   29.110
H   32.609   60.262   27.706
H   31.928   58.894   26.929
H   32.630   58.051   29.676
H   34.378   60.344   28.878
H   35.221   58.753   29.203
H   34.238   59.516   30.314
H   34.548   57.478   27.453
H   32.869   56.789   27.452
H   33.402   55.309   29.060
H   34.091   56.542   30.034
H   36.152   56.561   29.460
H   35.941   55.993   27.863
H   35.143   54.074   29.959
H   36.809   54.407   31.211
H   37.716   55.500   30.080
H   37.937   53.730   30.090
H   35.741   52.451   28.243
H   37.264   53.165   28.141
H   35.939   53.961   27.247
