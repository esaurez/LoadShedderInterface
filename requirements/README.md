# How to setup with Ansible 

## Executing ansible-playbook 
1. Install ansible following this [[link](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)].

2. Use following command to install:
```bash
ansible-playbook --ask-become-pass --inventory <ip_addr1>, ... <ip_addr2>, -u <user_name> ubuntu1804.yml 
```
  - Replace <ip_addr#> with ip address of the remote machine. Make the each ip address is seperated by comma(,) and put comma(,) after the last ip address
  - Replace <user_name> with the user name of the remote machine
