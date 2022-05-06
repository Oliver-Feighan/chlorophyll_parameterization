%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_501_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.510   58.908   41.214
C   26.389   57.486   40.173
C   31.078   56.056   39.826
C   32.447   60.512   41.462
C   27.841   61.854   41.884
N   28.827   56.959   40.121
C   27.518   56.700   39.825
C   27.436   55.532   38.855
C   28.911   54.976   39.030
C   29.715   56.028   39.694
C   29.081   53.624   39.801
C   26.963   56.031   37.452
C   25.856   55.222   36.659
C   26.197   54.602   35.292
O   26.908   53.636   35.228
O   25.743   55.328   34.211
N   31.439   58.349   40.916
C   31.911   57.134   40.353
C   33.362   57.111   40.338
C   33.747   58.430   40.814
C   32.511   59.138   41.090
C   34.114   55.981   39.769
C   35.215   58.861   40.977
O   36.088   58.038   40.574
C   35.695   60.196   41.585
N   30.146   61.001   41.363
C   31.386   61.370   41.621
C   31.404   62.847   41.849
C   29.909   63.220   42.036
C   29.214   61.950   41.652
C   32.316   63.462   42.995
C   29.401   64.488   41.174
C   28.934   65.756   41.930
N   27.563   59.512   41.266
C   27.032   60.713   41.551
C   25.619   60.696   41.446
C   25.330   59.423   40.949
C   26.560   58.732   40.808
C   24.611   61.727   41.761
C   24.219   58.634   40.514
O   23.054   58.953   40.433
C   24.922   57.260   39.910
C   24.424   56.085   40.635
O   24.275   56.006   41.861
O   24.194   55.107   39.691
C   23.683   53.831   40.189
C   26.344   54.828   32.981
C   26.274   56.032   32.101
C   26.623   56.061   30.771
C   26.994   54.774   29.954
C   26.468   57.426   30.002
C   27.690   57.801   29.109
C   27.775   59.238   28.732
C   29.265   59.656   28.482
C   29.270   61.038   27.720
C   30.062   59.758   29.785
C   31.457   58.964   29.818
C   31.522   57.713   30.726
C   32.136   56.378   30.085
C   31.189   55.139   30.125
C   33.417   56.050   30.777
C   34.222   54.877   30.123
C   35.486   55.439   29.522
C   36.258   54.499   28.468
C   37.746   54.252   28.902
C   36.203   55.087   26.983
H   31.620   55.201   39.419
H   33.359   61.112   41.460
H   27.253   62.676   42.298
H   26.787   54.792   39.324
H   29.195   54.980   37.977
H   29.758   53.810   40.636
H   29.533   52.827   39.211
H   28.178   53.232   40.270
H   27.912   55.977   36.919
H   26.652   57.074   37.397
H   25.025   55.920   36.559
H   25.423   54.489   37.339
H   33.711   55.819   38.769
H   33.897   55.071   40.328
H   35.203   56.026   39.798
H   35.472   60.989   40.871
H   36.689   59.969   41.969
H   35.053   60.303   42.460
H   31.808   63.295   40.942
H   29.828   63.268   43.122
H   32.558   64.463   42.637
H   33.189   62.822   43.122
H   31.692   63.520   43.887
H   28.595   64.076   40.567
H   30.174   64.712   40.439
H   27.969   66.123   41.583
H   29.691   66.531   41.807
H   28.891   65.630   43.012
H   23.617   61.297   41.880
H   24.715   62.312   40.847
H   25.045   62.408   42.493
H   24.835   57.215   38.825
H   22.808   53.967   40.825
H   24.471   53.304   40.729
H   23.372   53.225   39.338
H   25.703   54.160   32.404
H   27.372   54.471   33.037
H   25.928   56.929   32.615
H   28.083   54.807   29.965
H   26.689   54.974   28.927
H   26.589   53.846   30.360
H   26.285   58.237   30.708
H   25.570   57.353   29.389
H   27.404   57.235   28.223
H   28.495   57.380   29.712
H   27.298   59.862   29.487
H   27.131   59.252   27.853
H   29.656   58.959   27.740
H   28.346   61.269   27.191
H   30.040   60.911   26.960
H   29.423   61.894   28.378
H   29.449   59.490   30.646
H   30.351   60.784   30.014
H   32.165   59.689   30.219
H   31.710   58.735   28.783
H   30.550   57.441   31.138
H   32.145   58.020   31.566
H   32.336   56.385   29.014
H   31.240   54.626   29.164
H   30.225   55.491   30.493
H   31.507   54.329   30.781
H   33.153   55.809   31.807
H   34.065   56.925   30.726
H   33.675   54.331   29.354
H   34.559   54.070   30.773
H   36.153   55.330   30.377
H   35.250   56.403   29.072
H   35.735   53.552   28.333
H   37.713   53.544   29.730
H   38.165   55.219   29.180
H   38.356   53.691   28.193
H   35.440   54.493   26.481
H   37.129   54.922   26.432
H   36.036   56.164   26.955

