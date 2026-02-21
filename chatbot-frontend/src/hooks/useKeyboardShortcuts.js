import { useEffect, useCallback } from 'react';

export const useKeyboardShortcuts = ({
    onNewChat,
    onFocusInput,
    onToggleSidebar,
    onToggleSettings,
    onToggleTheme,
    onClearChat,
    inputRef
}) => {
    const handleKeyPress = useCallback((event) => {
        // Check if user is typing in an input field or textarea
        const isInputActive = event.target.tagName === 'INPUT' || 
                             event.target.tagName === 'TEXTAREA' || 
                             event.target.contentEditable === 'true';

        // Ctrl/Cmd + specific keys (work globally)
        if (event.ctrlKey || event.metaKey) {
            switch (event.key.toLowerCase()) {
                case 'n':
                    event.preventDefault();
                    onNewChat?.();
                    break;
                case 'k':
                    event.preventDefault();
                    onFocusInput?.();
                    break;
                case 'b':
                    event.preventDefault();
                    onToggleSidebar?.();
                    break;
                case ',':
                    event.preventDefault();
                    onToggleSettings?.();
                    break;
                case 'd':
                    event.preventDefault();
                    onToggleTheme?.();
                    break;
                case 'l':
                    event.preventDefault();
                    onClearChat?.();
                    break;
                default:
                    break;
            }
            return;
        }

        // Single key shortcuts (only when not typing)
        if (!isInputActive) {
            switch (event.key) {
                case '/':
                    event.preventDefault();
                    onFocusInput?.();
                    break;
                case 'Escape':
                    // Blur any focused input
                    if (document.activeElement) {
                        document.activeElement.blur();
                    }
                    break;
                default:
                    break;
            }
        }

        // Special case for Enter in chat input
        if (event.key === 'Enter' && event.target === inputRef?.current) {
            if (event.shiftKey) {
                // Shift + Enter = new line (default behavior)
                return;
            } else {
                // Enter = send message (handled by the input component)
                return;
            }
        }
    }, [onNewChat, onFocusInput, onToggleSidebar, onToggleSettings, onToggleTheme, onClearChat, inputRef]);

    useEffect(() => {
        document.addEventListener('keydown', handleKeyPress);
        return () => {
            document.removeEventListener('keydown', handleKeyPress);
        };
    }, [handleKeyPress]);

    return {
        shortcuts: [
            { key: 'Ctrl/Cmd + N', description: 'Start new chat' },
            { key: 'Ctrl/Cmd + K', description: 'Focus chat input' },
            { key: 'Ctrl/Cmd + B', description: 'Toggle sidebar' },
            { key: 'Ctrl/Cmd + ,', description: 'Open settings' },
            { key: 'Ctrl/Cmd + D', description: 'Toggle theme' },
            { key: 'Ctrl/Cmd + L', description: 'Clear current chat' },
            { key: '/', description: 'Focus chat input' },
            { key: 'Esc', description: 'Blur focused element' },
            { key: 'Enter', description: 'Send message' },
            { key: 'Shift + Enter', description: 'New line in message' }
        ]
    };
};
