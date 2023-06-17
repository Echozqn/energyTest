from energy.preTest.launcher import launcher

ps_hosts = launcher.getDemandInstance(['g4dn.xlarge'],[1], "../launcher/instancesInfo_demand.txt")
# ps_hosts = launcher.getDemandInstance(['p3.2xlarge'],[1], "../launcher/instancesInfo_demand.txt")
print(ps_hosts)
