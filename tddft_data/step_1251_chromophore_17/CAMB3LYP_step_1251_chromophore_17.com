%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1251_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.230   59.120   41.861
C   26.009   57.995   40.855
C   30.628   56.366   40.777
C   32.076   60.853   42.057
C   27.535   62.454   42.330
N   28.331   57.453   40.832
C   27.031   57.115   40.560
C   26.976   55.800   39.897
C   28.343   55.146   40.290
C   29.214   56.399   40.640
C   28.361   54.054   41.408
C   26.724   56.014   38.336
C   25.948   55.025   37.538
C   25.105   55.554   36.370
O   23.886   55.682   36.440
O   25.948   56.076   35.374
N   31.196   58.563   41.676
C   31.617   57.405   41.125
C   33.063   57.382   40.965
C   33.468   58.678   41.437
C   32.219   59.489   41.788
C   33.907   56.338   40.271
C   34.906   59.219   41.470
O   35.802   58.569   40.917
C   35.289   60.480   42.035
N   29.756   61.341   42.089
C   31.003   61.727   42.239
C   31.225   63.219   42.493
C   29.730   63.709   42.592
C   28.927   62.441   42.274
C   32.175   63.698   43.684
C   29.403   64.963   41.616
C   29.051   66.261   42.419
N   27.264   60.050   41.711
C   26.746   61.366   41.970
C   25.336   61.307   41.779
C   25.005   60.005   41.398
C   26.215   59.300   41.349
C   24.361   62.395   42.098
C   23.864   59.076   41.058
O   22.662   59.222   41.014
C   24.533   57.711   40.693
C   24.103   56.669   41.615
O   24.073   56.725   42.828
O   23.820   55.552   40.878
C   23.655   54.263   41.654
C   25.438   56.748   34.221
C   26.455   56.302   33.096
C   26.685   56.863   31.878
C   25.992   58.184   31.442
C   27.715   56.437   30.906
C   29.172   56.652   31.374
C   30.054   57.711   30.620
C   31.290   57.087   29.976
C   31.424   57.486   28.461
C   32.597   57.312   30.796
C   33.336   56.038   31.244
C   34.886   56.045   31.127
C   35.457   54.817   30.347
C   36.566   54.143   31.277
C   35.899   55.189   28.941
C   34.855   54.692   27.838
C   35.034   55.491   26.514
C   35.441   54.603   25.410
C   36.959   54.564   25.413
C   34.959   55.211   24.023
H   30.962   55.362   40.510
H   32.952   61.492   41.925
H   27.007   63.379   42.569
H   26.158   55.186   40.274
H   28.830   54.697   39.425
H   27.377   54.086   41.875
H   29.114   54.020   42.195
H   28.511   53.084   40.934
H   27.729   56.134   37.930
H   26.093   56.902   38.321
H   25.289   54.357   38.093
H   26.686   54.382   37.059
H   34.768   56.866   39.862
H   33.347   55.858   39.468
H   34.205   55.591   41.007
H   35.157   61.302   41.332
H   36.324   60.501   42.378
H   34.804   60.723   42.981
H   31.701   63.597   41.588
H   29.431   63.911   43.620
H   31.529   64.045   44.491
H   32.752   64.560   43.350
H   32.823   62.875   43.987
H   28.603   64.743   40.908
H   30.314   65.284   41.111
H   27.975   66.389   42.307
H   29.567   67.108   41.968
H   29.338   66.284   43.471
H   24.501   62.647   43.150
H   23.400   61.881   42.058
H   24.454   63.326   41.539
H   24.360   57.318   39.692
H   23.077   53.566   41.048
H   23.269   54.367   42.668
H   24.658   53.924   41.914
H   25.495   57.828   34.362
H   24.467   56.355   33.920
H   27.025   55.389   33.271
H   25.447   58.626   32.276
H   25.378   57.940   30.575
H   26.741   58.963   31.299
H   27.570   57.073   30.033
H   27.595   55.426   30.519
H   29.588   55.645   31.413
H   28.945   56.992   32.384
H   30.303   58.601   31.198
H   29.353   58.164   29.920
H   31.002   56.036   29.968
H   30.909   56.771   27.820
H   32.476   57.466   28.177
H   31.147   58.515   28.231
H   32.487   57.869   31.727
H   33.358   57.911   30.295
H   32.995   55.209   30.623
H   33.111   55.775   32.277
H   35.259   55.916   32.143
H   35.182   57.009   30.714
H   34.740   54.018   30.161
H   36.025   53.628   32.072
H   37.206   54.916   31.703
H   37.174   53.476   30.667
H   36.872   54.723   28.788
H   36.191   56.233   28.823
H   33.858   54.814   28.260
H   35.042   53.663   27.533
H   35.832   56.229   26.604
H   34.074   55.943   26.266
H   35.000   53.608   25.469
H   37.261   54.130   24.460
H   37.358   53.924   26.199
H   37.395   55.557   25.525
H   34.312   54.667   23.334
H   35.805   55.479   23.391
H   34.320   56.061   24.263

