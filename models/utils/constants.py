DATASETS = ['adult','covertype','arizona','vermont','geonames']
TRIGGER = ['same','random','near', 'min', 'interpolate']
CRITERIA = ['jaccard', 'dis', 'dif', 'randomflip', 'randomflipNN', 'randomgenNN', 'jaccardflipNN']
LABEL_STRATEGY = ['random', 'max', 'by_criteria', 'major']
PIVOT_METHOD = ['random', 'maxfreq', 'kpp']
MISSING_CODE = {'adult':['?']*15,
                'arizona':[9,0,0,9,9,9,0,0,[0,8,9],0,0,[0,9],9,0,9,[0,9],0,9999,[0,998,999],999,9,0,9,[0,8],0,0,0,[0,8],9,[0,9],0,9,[0,9],[0,9],[0,9],[0,9],[0,9],[0,9],[7,9],999,999,9999,9999,0,0,0,9999,[0,97,99],0,0,9999,0,[0,97,99],99,99,[98,99],[0,99],999,999,0,9999,0,[0,99999],99999,0,0,[0,997,999],999,[0,997,998,999],0,[998,999],9999,999,0,[0,999],999,9999,[997,998,999],999,[997,999],[997,998,999],999,9999,9999,9999,9999,[0,9999],[0,9999],[0,9999],9999,9999,[0,1,9999],9999,999999,999999,999999,[0,9999998,9999999],[999998,999999] ],
                'vermont':[0,9999,9,9,[0,8,9],9,9,9,0,0,0,0,0,0,[0,9],0,[0,9],9,999,0,9999,0,9,9,0,0,9,[0,8],[0,8],0,0,9,[0,9],[0,9],[0,9],[0,9],[0,9],[0,9],[7,9],[0,9],[0,998,999],999,9999,9999,0,[0,97,99],0,0,9999,0,0,9999,[0,97,99],99,[98,99],[0,99],99,0,999,999,0,[0,99999],99999,0,999,9999,9999,0,[0,997,999],[0,997,998,999],999,[998,999],0,9999,999,999,[997,999],999,[997,998,999],[0,999],0,[997,998,999],999,9999,9999,[0,1,9999],9999,[0,9999],9999,[0,9999],9999,[0,9999],9999,999999,999999,999999,[0,9999998,9999999],[999998,999999] ]
               }
