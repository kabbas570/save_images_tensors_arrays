

Best_DS = 0

all_sc = [2,6,4,8,9,8,7,6,8,4]

for epoch in range(10):
    
    sc = all_sc[epoch]
    if sc>Best_DS:
        Best_DS = sc
        

print(Best_DS)
    
