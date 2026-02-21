import { X, Keyboard } from 'lucide-react';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

export default function KeyboardShortcutsModal({ isOpen, onClose }) {
    const { shortcuts } = useKeyboardShortcuts({});

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-[hsl(var(--background))] rounded-lg shadow-xl max-w-md w-full max-h-[80vh] overflow-hidden border border-[hsl(var(--border))]">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-[hsl(var(--border))]">
                    <div className="flex items-center gap-2">
                        <Keyboard className="w-5 h-5 text-[hsl(var(--primary))]" />
                        <h2 className="text-lg font-semibold text-[hsl(var(--foreground))]">
                            Keyboard Shortcuts
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-md hover:bg-[hsl(var(--accent))] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Shortcuts List */}
                <div className="p-4 overflow-y-auto">
                    <div className="space-y-3">
                        {shortcuts.map((shortcut, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between py-2 px-3 rounded-lg bg-[hsl(var(--secondary))/50] hover:bg-[hsl(var(--secondary))] transition-colors"
                            >
                                <span className="text-sm text-[hsl(var(--foreground))]">
                                    {shortcut.description}
                                </span>
                                <kbd className="px-2 py-1 text-xs font-mono bg-[hsl(var(--accent))] border border-[hsl(var(--border))] rounded text-[hsl(var(--muted-foreground))]">
                                    {shortcut.key}
                                </kbd>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-[hsl(var(--border))] bg-[hsl(var(--secondary))/30]">
                    <p className="text-xs text-[hsl(var(--muted-foreground))] text-center">
                        Press <kbd className="px-1 py-0.5 text-xs font-mono bg-[hsl(var(--accent))] border border-[hsl(var(--border))] rounded">?</kbd> anytime to view shortcuts
                    </p>
                </div>
            </div>
        </div>
    );
}
