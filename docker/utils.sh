add_aliases() {
  # Write some useful aliases to "~/.bash_aliases"
  echo "alias ll='ls -lah'\n" >> "$HOME/.bash_aliases"
}

improve_prompt() {
  # Different prompt colours for root and non-root users
  if [ -z "$USER" ]
  then
    local USER_MODE="03"
    local USER_COLOR="36"
  else
    local USER_MODE="01"
    local USER_COLOR="33"
  fi

  # Define parts of PS1
  local USER_STR="\[\e[${USER_MODE};${USER_COLOR}m\]\u\[\e[00m\]"
  local WORKDIR_STR="\[\e[01;34m\]\w\[\e[00m\]"
  local GIT_STR="\[\e[0;35m\]\$(parse_git_branch)\[\e[00m\]"

  # Write configuration to ~/.bashrc
  echo "
function parse_git_branch {
  local ref
  ref=\$(command git symbolic-ref HEAD 2> /dev/null) || return 0
  echo \"‹\${ref#refs/heads/}› \"
}

PS1='${USER_STR} :: ${WORKDIR_STR} ${GIT_STR}$ '
" >> "$HOME/.bashrc"
}

configure_jupyter() {
# Configure Jupyter
# Set --no-browser --ip=0.0.0.0
  jupyter lab --generate-config
  echo 'c.ServerApp.ip = "0.0.0.0"' >> "$HOME/.jupyter/jupyter_lab_config.py"
  echo 'c.ExtensionApp.open_browser = False' >> "$HOME/.jupyter/jupyter_lab_config.py"
}

configure_user() {
  # If this directory doesn't exist it won't be included in the $PATH
  # and python entrypoints for user-installed packages won't work
  mkdir -p "$HOME/.local/bin"

  # miscellaneous tweaks and settings
  add_aliases
  improve_prompt
  configure_jupyter
}

create_users() {
  local USERS="$1"
  local GROUP="$2"

  for x in $(echo "$USERS" | tr "," "\n")
  do
    # skip empty user entries
    if [ -z "$x" ]
    then
      continue
    fi

    # Create and configure user
    user_name="${x%/*}"
    user_id="${x#*/}"
    user_home="/home/${user_name}"
    useradd --create-home --uid "$user_id" --gid "$GROUP" --home-dir "$user_home" "$user_name"
    su "$user_name" -c "$(declare -f configure_user add_aliases improve_prompt configure_jupyter) &&  configure_user"
    echo "Added user ${user_name} with ID ${user_id}"
  done
}
