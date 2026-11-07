FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# A imagem base traz o repositório apt do Yarn com chave GPG inválida
# (NO_PUBKEY 62D54FD4003F6525), o que faz qualquer `apt-get update` falhar
# e quebra a instalação de features (ex.: sshd). O projeto não usa Yarn.
RUN rm -f /etc/apt/sources.list.d/yarn.list
