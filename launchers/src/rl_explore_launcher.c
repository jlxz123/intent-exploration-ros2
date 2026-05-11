#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef RL_EXPLORE_LAUNCH_SCRIPT
#define RL_EXPLORE_LAUNCH_SCRIPT "scripts/run_rl_explore_all.sh"
#endif

#ifndef RL_EXPLORE_LAUNCHER_NAME
#define RL_EXPLORE_LAUNCHER_NAME "RL Explore"
#endif

static void log_error(const char *message, const char *detail)
{
    FILE *fp = fopen("/tmp/rl_explore_launcher_error.log", "a");
    if (fp == NULL) {
        return;
    }
    fprintf(fp, "[%s] %s", RL_EXPLORE_LAUNCHER_NAME, message);
    if (detail != NULL && detail[0] != '\0') {
        fprintf(fp, ": %s", detail);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

static int strip_last_component(char *path)
{
    char *slash = strrchr(path, '/');
    if (slash == NULL) {
        return -1;
    }
    if (slash == path) {
        slash[1] = '\0';
    } else {
        *slash = '\0';
    }
    return 0;
}

int main(int argc, char **argv)
{
    char exe_path[PATH_MAX];
    ssize_t read_len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (read_len < 0) {
        log_error("failed to resolve /proc/self/exe", strerror(errno));
        return 1;
    }
    exe_path[read_len] = '\0';

    char launchers_dir[PATH_MAX];
    snprintf(launchers_dir, sizeof(launchers_dir), "%s", exe_path);
    if (strip_last_component(launchers_dir) != 0) {
        log_error("failed to resolve launcher directory", exe_path);
        return 1;
    }

    char project_root[PATH_MAX];
    snprintf(project_root, sizeof(project_root), "%s", launchers_dir);
    if (strip_last_component(project_root) != 0) {
        log_error("failed to resolve project root", launchers_dir);
        return 1;
    }

    char script_path[PATH_MAX];
    int written = snprintf(script_path, sizeof(script_path), "%s/%s", project_root, RL_EXPLORE_LAUNCH_SCRIPT);
    if (written < 0 || (size_t)written >= sizeof(script_path)) {
        log_error("script path is too long", RL_EXPLORE_LAUNCH_SCRIPT);
        return 1;
    }

    if (access(script_path, R_OK) != 0) {
        log_error("launch script is not readable", script_path);
        return 1;
    }

    if (chdir(project_root) != 0) {
        log_error("failed to enter project root", strerror(errno));
        return 1;
    }

    char **child_argv = calloc((size_t)argc + 2, sizeof(char *));
    if (child_argv == NULL) {
        log_error("failed to allocate argv", strerror(errno));
        return 1;
    }

    child_argv[0] = "bash";
    child_argv[1] = script_path;
    for (int i = 1; i < argc; ++i) {
        child_argv[i + 1] = argv[i];
    }
    child_argv[argc + 1] = NULL;

    execvp("bash", child_argv);
    log_error("failed to execute launch script", strerror(errno));
    free(child_argv);
    return 1;
}
