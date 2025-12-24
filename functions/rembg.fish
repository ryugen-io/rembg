function rembg --description 'Background removal for game sprites'
    # Check if rembg is installed (pipx)
    if not command -v rembg &>/dev/null
        echo "Error: rembg not installed"
        echo "Run: cd ~/code/github.com/ryugen-io/rembg && ./install.sh"
        return 1
    end

    # Show usage if no args
    if test (count $argv) -lt 2
        echo "Usage: rembg <mode> <files...> [passes]"
        echo ""
        echo "Modes:"
        echo "  pixel       Pixel art (tight tolerances, no edge cleanup)"
        echo "  remove      Standard mode (balanced tolerances + edge cleanup)"
        echo "  aggressive  Post-processing cleanup (YCbCr distance-based)"
        echo ""
        echo "Optional:"
        echo "  [passes]    Number of cleanup passes (1-10, default: 1)"
        echo "              Only for 'remove' mode"
        echo ""
        echo "Examples:"
        echo "  rembg pixel retro-sprite.png      # Pixel art, 1 pass (default)"
        echo "  rembg remove GOTENKS.png 5        # Standard mode, 5 passes"
        echo "  rembg aggressive sprite_transparent.png  # Post-processing cleanup"
        return 1
    end

    set -l command $argv[1]
    set -l passes 1
    set -l files $argv[2..-1]

    # Check if last argument is a number (passes)
    set -l last_arg $argv[-1]
    if string match -qr '^[0-9]+$' $last_arg
        set passes $last_arg
        # Remove passes from files list
        set files $argv[2..-2]

        # Validate passes range
        if test $passes -lt 1 -o $passes -gt 10
            echo "Error: passes must be between 1 and 10"
            return 1
        end

        if test (count $files) -eq 0
            echo "Error: No files specified"
            return 1
        end
    end

    # Auto-resolve files: if just a filename (no path), search in dev/assets/
    set -l resolved_files
    for file in $files
        # Check if file is just a filename (no directory separators)
        if not string match -q '*/*' $file
            # For cleanup mode, check proc/ first (for *_transparent.png)
            # For other modes, check raw/
            set -l found_file ""
            if test "$command" = "cleanup"
                # Check proc/ for transparent files
                if test -f $project_dir/dev/assets/proc/$file
                    set found_file $project_dir/dev/assets/proc/$file
                    echo "Found: $file → dev/assets/proc/$file"
                end
            end

            # If not found yet, check raw/
            if test -z "$found_file"
                if test -f $project_dir/dev/assets/raw/$file
                    set found_file $project_dir/dev/assets/raw/$file
                    echo "Found: $file → dev/assets/raw/$file"
                end
            end

            if test -n "$found_file"
                set -a resolved_files $found_file
            else
                echo "Error: File not found in dev/assets/raw/ or dev/assets/proc/: $file"
                return 1
            end
        else
            # File has path, use as-is
            set -a resolved_files $file
        end
    end

    switch $command
        case pixel
            command rembg $resolved_files --autocrop --pixel-art --passes $passes
        case remove
            command rembg $resolved_files --autocrop --passes $passes
        case aggressive
            command rembg $resolved_files --autocrop --use-grabcut
        case '*'
            echo "Error: Unknown mode '$command'"
            echo "Valid modes: pixel, remove, aggressive"
            return 1
    end
end
