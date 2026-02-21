import React, { useState, useEffect } from 'react';
import { ChevronDown, Globe, Cpu, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const ChatSettings = ({ 
    selectedModel, 
    onModelChange, 
    useWebSearch, 
    onWebSearchToggle, 
    availableModels, 
    isLoading 
}) => {
    const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
    const { theme, toggleTheme } = useTheme();

    return (
        <div className="flex items-center justify-between p-3 bg-[hsl(var(--secondary))] border-b border-[hsl(var(--border))]">
            <div className="flex items-center space-x-4">
                {/* Model Selection Dropdown */}
                <div className="relative">
                    <button
                        onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                        className="flex items-center space-x-2 px-3 py-2 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg hover:bg-[hsl(var(--accent))] transition-colors"
                        disabled={isLoading}
                    >
                        <Cpu className="w-4 h-4 text-[hsl(var(--muted-foreground))]" />
                        <span className="text-sm font-medium text-[hsl(var(--foreground))]">
                            {selectedModel.replace('models/', '').replace('/', ' / ')}
                        </span>
                        <ChevronDown className={`w-4 h-4 text-[hsl(var(--muted-foreground))] transition-transform ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {isModelDropdownOpen && (
                        <div className="absolute top-full left-0 mt-1 w-80 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
                            {Object.entries(availableModels).map(([provider, models]) => (
                                <div key={provider} className="p-2">
                                    <h3 className="text-xs font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wide mb-2 px-2">
                                        {provider}
                                    </h3>
                                    {models.map((model) => (
                                        <button
                                            key={model.name}
                                            onClick={() => {
                                                onModelChange(model.name);
                                                setIsModelDropdownOpen(false);
                                            }}
                                            className={`w-full text-left px-3 py-2 rounded-md text-sm hover:bg-[hsl(var(--accent))] transition-colors ${
                                                selectedModel === model.name ? 'bg-[hsl(var(--primary))/10] text-[hsl(var(--primary))]' : 'text-[hsl(var(--foreground))]'
                                            }`}
                                        >
                                            {model.display_name}
                                        </button>
                                    ))}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Web Search Toggle */}
                <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={useWebSearch}
                        onChange={(e) => onWebSearchToggle(e.target.checked)}
                        className="sr-only"
                        disabled={isLoading}
                    />
                    <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                        useWebSearch 
                            ? 'bg-[hsl(var(--primary))/10] text-[hsl(var(--primary))] border border-[hsl(var(--primary))]' 
                            : 'bg-[hsl(var(--background))] text-[hsl(var(--muted-foreground))] border border-[hsl(var(--border))] hover:bg-[hsl(var(--accent))]'
                    }`}>
                        <Globe className="w-4 h-4" />
                        <span className="text-sm font-medium">Web Search</span>
                    </div>
                </label>
            </div>

            {/* Theme Toggle Button */}
            <button
                onClick={toggleTheme}
                className="flex items-center space-x-2 px-3 py-2 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg hover:bg-[hsl(var(--accent))] transition-colors"
                title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
                disabled={isLoading}
            >
                {theme === 'light' ? (
                    <Moon className="w-4 h-4 text-[hsl(var(--muted-foreground))]" />
                ) : (
                    <Sun className="w-4 h-4 text-[hsl(var(--muted-foreground))]" />
                )}
                <span className="text-sm font-medium text-[hsl(var(--foreground))]">
                    {theme === 'light' ? 'Dark' : 'Light'}
                </span>
            </button>

            {/* Close dropdown when clicking outside */}
            {isModelDropdownOpen && (
                <div 
                    className="fixed inset-0 z-40"
                    onClick={() => setIsModelDropdownOpen(false)}
                />
            )}
        </div>
    );
};

export default ChatSettings;
