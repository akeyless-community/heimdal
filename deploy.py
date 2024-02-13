from heimdal.tools.utility_tools.kubernetes_operations import can_i_deploy_into_namespace


result = can_i_deploy_into_namespace("python")
print(f"Can I deploy into the namespace? {result}")
