import pandas as pd
import numpy as np

np.random.seed(0)

data = pd.read_csv('DSL-StrongPasswordData.csv')

# normalize
intervals = data.keys()[3:]
data[intervals]=data[intervals].div(data[intervals].sum(axis=1), axis=0)


subs = data['subject'].unique()

def get_sign(sub):
    samples = data[data['subject']==sub].sample(10)
    samples = samples.drop(['rep', 'sessionIndex', 'subject'], axis=1)

    return samples.mean()

all_signs = dict()

for sub in subs:

    all_signs[sub] = get_sign(sub)

# keystroke detector (part 2)
def get_match(sample, sign):

    dist = np.sqrt((((sample-sign))**2).sum())
    # return 1 when dist is zero, otherwise <1
    return 1/(1+2*dist)

def get_sucess_percent(samples, sign):

    total_count = len(samples)
    success_count = 0

    for i, sample in samples.iterrows():

        match = get_match(sample, sign)
        
        if match>0.85:
            success_count += 1

    return success_count/total_count

all_success_percent = {}
sheep_count = 0
goat_count = 0

frr = []
far = []

for sub in subs:

    sp = get_sucess_percent(data[data['subject']==sub][intervals], all_signs[sub])
    # taking all takes too much time, so we sample 1000
    sp2 = get_sucess_percent(data[data['subject']!=sub][intervals].sample(1000), all_signs[sub])
    if sp>0.7:
        sub_type = 'Sheep'
        sheep_count += 1
    else:
        sub_type = 'Goat'
        goat_count += 1


    print(sub, sp, sub_type)
    print("FRR for", sub,"=",1-sp)
    frr.append(1-sp)
    print("FAR for", sub,"=",sp2)
    far.append(sp2)

print('Sheep count: ', sheep_count)
print('Goat count: ', goat_count)
print('Overall FRR: ', np.mean(frr))
print('Overall FAR: ', np.mean(far))

print()


all_imitation_counts = {}

for sub1 in subs:

    sub2_count1 = 0
    sub2_count2 = 0
    # taking all takes too much time. Hence we take 25 random samples
    sub1_data = data[data['subject']==sub1][intervals].sample(100)

    for sub2 in subs:

        if sub2==sub1:
            continue

        sp1 = get_sucess_percent(sub1_data, all_signs[sub2])
        sp2 = get_sucess_percent(data[data['subject']==sub2][intervals].sample(100), all_signs[sub1])

        # if 70% attempts are successful, we say that imitation is possible 
        if sp1>0.7:
            sub2_count1 += 1
        if sp2>0.7:
            sub2_count2 += 1

    all_imitation_counts[sub1] = [sub2_count1, sub2_count2]
    print(sub1, "can imitate", sub2_count1, "other subjects")
    print(sub1, "can be imitated by", sub2_count2, "other subjects")


for sub in subs:

    can_imitate = all_imitation_counts[sub][0]
    imitated_by = all_imitation_counts[sub][1]

    if can_imitate>5: # if subject can imitate 5 other subjects
        print(sub, "is Wolf")
    if imitated_by>5: # if subject is imitated by 5 other subjects
        print(sub, "is Lamb")


# initialization: from file (generated randomly)
pop = pd.read_csv('random_num1.csv')
pop = pop[intervals].div(pop[intervals].sum(axis=1), axis=0)
pop_len = len(pop)

attacked_sub = 's002'
n_parents = pop_len//2
p_mutate = 0.25

def get_child(parents):

    low = parents.min()
    high = parents.max()

    child = parents.iloc[0].copy()

    for col in intervals:
        # crossover: each value chosen uniformly at random between parent's values
        child[col] = np.random.uniform(low[col], high[col])

    # mutation: occurs with prob 0.25
    if np.random.binomial(1, p_mutate)==1:
        # mutation: randomly selected column is assigned a value randomly within a period with length thrice of parent's interval
        col = np.random.choice(intervals)
        diff = high[col]-low[col]
        child[col] = np.random.uniform(low[col]-diff, high[col]+diff)
    
    child = child.div(child.sum(), axis=0)

    return child


sign = all_signs[attacked_sub]

sp = get_sucess_percent(pop, sign)
gen = 0

# termination: when 70% population is successful (match>85%), we stop
while sp<0.7:
    gen += 1
    fitness = [0]*pop_len

    for i in range(pop_len):

        # print(i)
        fitness[i] = get_match(pop.iloc[i], sign)

    print("Generation:", gen)
    print("Success %:", sp)
    print("Max fitness:", max(fitness))
    print("Min fitness:", min(fitness))
    print("Avg fitness:", np.mean(fitness))
    print()
    # parents = pop.sample(n=n_parents, weights=fitness)
    pop['fitness'] = fitness
    # selection: top 50% survive and produce children untill population is restored 
    parents = pop.nlargest(n_parents, ['fitness'])[intervals]
    pop = parents.copy()

    while len(pop) < pop_len:

        # crossover: 2 randomly selected survivors mate to create child
        child = get_child(parents.sample(2))
        pop = pop.append(child)

    sp = get_sucess_percent(pop, sign)
    pop.reset_index(drop=True, inplace=True)



gen += 1
fitness = [0]*pop_len

for i in range(pop_len):

    # print(i)
    fitness[i] = get_match(pop.iloc[i], sign)

print("Generation:", gen)
print("Success %:", sp)
print("Max fitness:", max(fitness))
print("Min fitness:", min(fitness))
print("Avg fitness:", np.mean(fitness))
print()
