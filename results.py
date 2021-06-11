from predictingColleges import result
result = sorted(result.items(), key = lambda kv:(kv[1][0], kv[0][0]), reverse=True)

rankedColleges = []
for i in result :
    rankedColleges.append(i[0])
print(result)   
print(rankedColleges)

