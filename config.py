million = 1000000
minute = 60
hour = 60*60

perf_bench_keyspace = 3 * million

valkey_binary = 'valkey-server'

# ip address of dedicated server for building and running Valkey server
server = "172.31.70.82"

# ssh key ti use when accessing the server
sshkeyfile = 'server-keyfile.pem'

conductress_log = './log.txt'
conductress_output = './output.txt'
conductress_data_dump = './testdump.txt'
