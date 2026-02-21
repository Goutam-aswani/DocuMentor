import { useState, useEffect, useRef } from 'react';
import { Plus, LogOut, MessageSquare, Trash2, MoreHorizontal, Pencil, Check, X, ChevronsLeft,PanelRight,User, Sun, Moon, BarChart3 } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useTheme } from '../context/ThemeContext';
import { Link } from 'react-router-dom';
import { Settings } from 'lucide-react';

const SessionItem = ({ session, onSelect, onDelete, onRename, activeSessionId }) => {
    const [menuOpen, setMenuOpen] = useState(false);
    const [isRenaming, setIsRenaming] = useState(false);
    const [title, setTitle] = useState(session.title);
    const inputRef = useRef(null);

    useEffect(() => {
        if (isRenaming) {
            inputRef.current?.focus();
            inputRef.current?.select();
        }
    }, [isRenaming]);

    const handleDeleteClick = (e) => {
        e.stopPropagation();
        setMenuOpen(false);
        onDelete(session.id);
    };

    const handleRenameClick = (e) => {
        e.stopPropagation();
        setMenuOpen(false);
        setIsRenaming(true);
    };

    const handleSaveRename = (e) => {
        e.stopPropagation();
        if (title.trim()) {
            onRename(session.id, title.trim());
            setIsRenaming(false);
        }
    };

    const handleCancelRename = (e) => {
        e.stopPropagation();
        setTitle(session.title);
        setIsRenaming(false);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSaveRename(e);
        } else if (e.key === 'Escape') {
            handleCancelRename(e);
        }
    };

    const handleMenuToggle = (e) => {
        e.stopPropagation();
        setMenuOpen(prev => !prev);
    };

    return (
        // Use CSS variables for SessionItem container
        <div className="relative group bg-[hsl(var(--secondary))] hover:bg-[hsl(var(--secondary))/70] rounded-xl shadow-lg transition-colors duration-200 ease-in-out">
            {isRenaming ? (
                // Use CSS variables for renaming input container
                <div className="flex items-center w-full bg-[hsl(var(--accent))] rounded-lg border border-[hsl(var(--border))]">
                    <input
                        ref={inputRef}
                        type="text"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        onKeyDown={handleKeyDown}
                        // Use CSS variables for renaming input text
                        className="flex-grow bg-transparent px-3 py-2 text-sm text-[hsl(var(--foreground))] outline-none"
                    />
                    {/* Colors for rename/cancel buttons are kept as in your draft (adjust if needed for theming) */}
                    <button onClick={handleSaveRename} className="p-2 text-green-400 hover:text-white"><Check className="w-4 h-4" /></button>
                    <button onClick={handleCancelRename} className="p-2 text-red-400 hover:text-white"><X className="w-4 h-4" /></button>
                </div>
            ) : (
                <>
                    <button
                        onClick={() => onSelect(session.id)}
                        // Use CSS variables for session button
                        className={`flex items-center w-full text-left pl-3 pr-10 py-2.5 text-sm rounded-xl my-2 shadow-md truncate transition-colors duration-200 ease-in-out ${
                            activeSessionId === session.id
                            ? 'bg-[hsl(var(--primary))/10] text-[hsl(var(--primary))] border-l-2 border-[hsl(var(--primary))]'
                            : 'text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))/50]'
                        }`}
                    >
                        <MessageSquare className="w-4 h-4 mr-3 flex-shrink-0" />
                        <span className="truncate">{session.title}</span>
                    </button>
                    <button
                        onClick={handleMenuToggle}
                        // Use CSS variables for more options button
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-[hsl(var(--muted-foreground))] rounded-md bg-transparent opacity-0 group-hover:opacity-100 hover:text-[hsl(var(--foreground))]"
                        title="More options"
                    >
                        <MoreHorizontal className="w-4 h-4" />
                    </button>
                    {menuOpen && (
                        // Use CSS variables for menu dropdown
                        <div className="absolute z-10 right-0 mt-1 w-32 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-md shadow-lg">
                            <button onClick={handleRenameClick} className="flex items-center w-full px-3 py-2 text-sm text-left text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]">
                                <Pencil className="w-4 h-4 mr-2" /> Rename
                            </button>
                            {/* Colors for delete button are kept as in your draft (adjust if needed for theming) */}
                            <button onClick={handleDeleteClick} className="flex items-center w-full px-3 py-2 text-sm text-left text-red-500 hover:bg-red-600/20">
                                <Trash2 className="w-4 h-4 mr-2" /> Delete
                            </button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};


export default function Sidebar({ sessions, onSelectSession, onNewChat, onDeleteSession, onRenameSession, activeSessionId, isLoading, isCollapsed, onToggleCollapse }) {
    const { logout } = useAuth();
    const { theme, toggleTheme } = useTheme();

    return (
        // Use CSS variables for the main sidebar container
        <div className={`bg-[hsl(var(--background))] border-r border-[hsl(var(--border))] flex flex-col transition-all duration-300 ease-in-out ${isCollapsed ? 'w-16 p-2' : 'w-64 p-3'}`}>

            {isCollapsed ? (
                // --- COLLAPSED VIEW ---
                <>
                    {/* Use CSS variables for Expand Button */}
                    <button onClick={onToggleCollapse} className="p-1.5 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] mb-3 transition-colors duration-200 ease-in-out">
                        <PanelRight className="w-5 h-5" />
                    </button>
                    {/* New Chat Button (Icon Only) - Use CSS variables for colors */}
                    <button
                        onClick={onNewChat}
                        className="flex items-center justify-center w-full h-10 mb-3 text-[hsl(var(--primary-foreground))] bg-[hsl(var(--primary))] rounded-lg hover:bg-gradient-to-r hover:from-[hsl(var(--primary))] hover:to-blue-600 hover:shadow-lg transform hover:scale-105 transition-all duration-200 ease-in-out shadow-md"
                        title="New Chat"
                    >
                        <Plus className="w-5 h-5" />
                    </button>
                    {/* Spacer to push logout to the bottom */}
                    <div className="flex-grow"></div>
                    
                    {/* Theme Toggle Button (Icon Only) - Use CSS variables for colors */}
                    {/* <button
                        onClick={toggleTheme}
                        className="flex items-center justify-center w-full h-12 mb-2 text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--foreground))] transition-colors duration-200 ease-in-out border border-[hsl(var(--border))]"
                        title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
                    >
                        {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
                    </button> */}
                    
                    {/* Logout Button (Icon Only) - Use CSS variables for colors */}
                    <button
                        onClick={logout}
                        className="flex items-center justify-center w-full h-10 text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--destructive))/80] hover:text-[hsl(var(--destructive-foreground))] transition-colors duration-200 ease-in-out border border-[hsl(var(--border))]"
                        title="Logout"
                    >
                        <LogOut className="w-5 h-5" />
                    </button>
                </>
            ) : (
                // --- EXPANDED VIEW ---
                <>
                    {/* Profile Button - Use CSS variables */}
                    <div className="flex items-center justify-between mb-4">
                    <Link to = "/profile"
                    className="flex items-center text-sm font-semibold text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))/50] rounded-lg transition-colors p-3"
                    ><User className= "size-7"/>
                    <h2 className="ml-3 text-lg font-jp text-[hsl(var(--foreground))]">Profile</h2>
                    </Link>

                    {/* Collapse Button - Use CSS variables */}
                        <button onClick={onToggleCollapse} className="flex items-center text-sm font-semibold text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))/50] rounded-lg transition-colors p-3 ease-in-out">
                        <PanelRight className="size-6" />
                    </button>
                    </div>
                    {/* New Chat Button (With Text) - Use CSS variables for colors */}
                    <button
                        onClick={onNewChat}
                        className="flex items-center justify-center w-full px-4 py-2.5 mb-4 font-semibold text-[hsl(var(--primary-foreground))] bg-[hsl(var(--primary))] rounded-lg hover:bg-gradient-to-r hover:from-[hsl(var(--primary))] hover:to-blue-600 hover:shadow-lg transform hover:scale-105 transition-all duration-200 ease-in-out shadow-md"
                    >
                        <Plus className="w-4 h-4 mr-2" /> New Chat
                    </button>
                    {/* History List */}
                    <div className="flex-grow overflow-y-auto -mr-2 pr-2">
                        {/* Use CSS variables for History heading */}
                        <h2 className="text-xs font-bold tracking-wider text-[hsl(var(--muted-foreground))] uppercase mb-2 px-2">History</h2>
                        {isLoading ? (
                            
                            <p className="text-[hsl(var(--muted-foreground))] px-2">Loading...</p>
                        ) : (
                            sessions.map(session => (
                                <SessionItem
                                    key={session.id}
                                    session={session}
                                    onSelect={onSelectSession}
                                    onDelete={onDeleteSession}
                                    onRename={onRenameSession}
                                    activeSessionId={activeSessionId}
                                />
                            ))
                        )}
                    </div>

                    {/* Settings Button - Use CSS variables */}
<div className="pt-2 border-t border-[hsl(var(--border))]">
                        <Link
                            to="/usage"
                            className="flex items-center justify-center w-full px-4 py-2 mt-2 text-sm font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--foreground))] transition-colors border border-[hsl(var(--border))]"
                        >
                            <BarChart3 className="w-4 h-4 mr-2" /> Usage Stats
                        </Link>
                        
                        <Link
                            to="/settings"
                            className="flex items-center justify-center w-full px-4 py-2 mt-2 text-sm font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--foreground))] transition-colors border border-[hsl(var(--border))]"
                        >
                            <Settings className="w-4 h-4 mr-2" /> Settings
                        </Link>
                        
                        {/* Theme Toggle Button - Use CSS variables */}
                        {/* <button
                            onClick={toggleTheme}
                            className="flex items-center justify-center w-full px-4 py-2 mt-2 text-sm font-semibold text-[hsl(var(--muted-foreground))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--foreground))] transition-colors border border-[hsl(var(--border))]"
                        >
                            {theme === 'light' ? (
                                <>
                                    <Moon className="w-4 h-4 mr-2" /> Dark Mode
                                </>
                            ) : (
                                <>
                                    <Sun className="w-4 h-4 mr-2" /> Light Mode
                                </>
                            )}
                        </button> */}
                    </div>
                     {/* Logout Button (With Text) - Use CSS variables for colors */}
                    <button onClick={logout} className="flex items-center justify-center w-full px-4 py-2.5 mt-4 text-[hsl(var(--destructive))] bg-[hsl(var(--secondary))] rounded-lg hover:bg-[hsl(var(--destructive))/20] transition-colors shadow-md border border-[hsl(var(--border))]">
 <LogOut className="w-4 h-4 mr-2" /> Logout
 </button>
                </>
            )}
        </div>
    );
};