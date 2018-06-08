import pandas as pd
import numpy as np
import Read_Data as RD
from pomegranate import BayesianNetwork


def transfer(data, bounds, num_features, nominal_feature):
    size = data.shape[0]
    for k in range(size):
        initial_sample = data[k, :]
        z = np.linspace(0, 0, num_features)
        for ii in range(num_features):
            if ii not in nominal_feature:
                z[ii] = np.max(np.where(bounds[:, ii] <= initial_sample[ii])[0])
                if z[ii] > 99:
                    z[ii] -= 1
            else:
                z[ii] = initial_sample[ii]
        data[k, :] = z


#file = 'poker-8-9_vs_5.dat'
file = 'kddcup-rootkit-imap_vs_back.dat'
name = file.split('.')[0]
print(name)

#nominal_index = [0, 2, 4]
#nominal_value = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
#nominal_index = [0]
#nominal_value = ['M', 'F', 'I']

nominal_index = [1,2,3]
nominal_value = [['icmp', 'tcp', 'udp'],
                 ['auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                  'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'hostnames',
                  'http', 'http_443', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login',
                  'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
                  'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
                  'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'time', 'tim_i', 'urh_i',
                  'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'],
                 ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']]


#RD.Initialize_Data(file)
RD.Initialize_Data(file, has_nominal=True, nominal_index=nominal_index, nominal_value=nominal_value)
print('Number of Positive: ', RD.Num_positive)
print('Number of Negative: ', RD.Num_negative)

nominal_feature = [1,2,3,6,7,8,10,11,13,14,17,18,19,20,21]
#nominal_feature = [0,1,2,3,4,5,6,7,8,9]
data = RD.get_feature()
num_samples = data.shape[0]
num_features = data.shape[1]
num_bins = 100
bounds = np.zeros((num_bins+1, num_features))
for i in range(num_features):
    if i not in nominal_feature:
        bounds[:, i] = np.histogram(data[:, i], bins=num_bins)[1]

nf = RD.get_negative_feature()
transfer(nf, bounds, num_features, nominal_feature)
pf = RD.get_positive_feature()
transfer(pf, bounds, num_features, nominal_feature)
np.savez(name+'.npz', N_F=nf, P_F=pf)

'''
bayes = BayesianNetwork.from_samples(nf, algorithm='exact-dp')
pt = bayes.log_probability(nf).sum()
print('Exact Shortest:', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='exact')
pt = bayes.log_probability(nf).sum()
print('Exact A*', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='greedy')
pt = bayes.log_probability(nf).sum()
print('Greedy', pt)


bayes = BayesianNetwork.from_samples(nf, algorithm='chow-liu')
pt = bayes.log_probability(nf).sum()
print('Chow-Liu', pt)

with open(name+'_bayes.json', 'w') as w:
    w.write(bayes.to_json())
'''






