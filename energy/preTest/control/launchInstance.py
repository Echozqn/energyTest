from energy.preTest.launcher import launcher

ps_hosts = launcher.getDemandInstance(['g4dn.xlarge'],[2], "../launcher/instancesInfo_demand.txt")
print(ps_hosts)
