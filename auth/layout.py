# auth/layout.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import dash_bootstrap_components as dbc
from dash import html, dcc

# Define common color palette to maintain consistency
COLORS = {
    'primary': '#6b5b95',      # Main violet
    'secondary': '#b8b8d1',    # Lighter violet
    'background': '#f8f7fc',   # Very light violet background
    'text': '#2d283e',         # Dark violet text
    'accent': '#8677aa',       # Mid-tone violet
    'light_accent': '#e6e4ed', # Very light violet for borders
    'white': '#ffffff',        # Pure white
    'shadow': '0 4px 12px rgba(107, 91, 149, 0.12)' # Violet-tinted shadow
}

def create_login_layout():
    """Create an enhanced login layout with improved visuals"""
    return html.Div([
        # Background container with gradient
        html.Div(
            style={
                'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["white"]} 100%)',
                'minHeight': '100vh',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'padding': '2rem 1rem'
            },
            children=[
                # Content container
                dbc.Container([
                    dbc.Row([
                        # Left column with brand/logo (for medium screens and up)
                        dbc.Col([
                            html.Div([
                                html.I(
                                    className="fas fa-robot fa-4x mb-4",
                                    style={'color': COLORS['primary']}
                                ),
                                html.H1(
                                    "IMDA Smart Viewer",
                                    className="mb-3",
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': '700',
                                        'fontSize': '2.5rem'
                                    }
                                ),
                                html.P(
                                    "Upload documents and ask questions across all of them using advanced AI technology.",
                                    style={
                                        'color': COLORS['text'],
                                        'fontSize': '1.1rem',
                                        'opacity': '0.85',
                                        'maxWidth': '400px'
                                    }
                                ),
                                html.Hr(style={
                                    'width': '50px', 
                                    'border': f'2px solid {COLORS["secondary"]}',
                                    'opacity': '1',
                                    'margin': '2rem 0'
                                }),
                                html.P(
                                    "Access your documents securely and get insights with our powerful AI assistant.",
                                    style={
                                        'color': COLORS['text'],
                                        'opacity': '0.7',
                                        'fontSize': '0.95rem'
                                    }
                                )
                            ], style={
                                'height': '100%',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'justifyContent': 'center',
                                'padding': '2rem'
                            })
                        ], md=6, className="d-none d-md-block"),
                        
                        # Right column with login form
                        dbc.Col([
                            # Card with login form
                            dbc.Card([
                                dbc.CardBody([
                                    # Header
                                    html.Div([
                                        html.H2(
                                            "Welcome Back",
                                            className="text-center mb-1",
                                            style={
                                                'fontWeight': '600',
                                                'color': COLORS['text']
                                            }
                                        ),
                                        html.P(
                                            "Sign in to continue to IMDA Smart Viewer",
                                            className="text-center mb-4",
                                            style={
                                                'color': COLORS['text'],
                                                'opacity': '0.7'
                                            }
                                        ),
                                    ]),
                                    
                                    # Login form
                                    html.Div([
                                        # Username field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-user"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="login-username",
                                                type="text",
                                                placeholder="Username",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-3"),
                                        
                                        # Password field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-lock"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="login-password",
                                                type="password",
                                                placeholder="Password",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-4"),
                                        
                                        # Remember me and forgot password row
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Checkbox(
                                                    id="remember-me",
                                                    label="Remember me",
                                                    className="custom-control-input",
                                                ),
                                                width="auto"
                                            ),
                                            dbc.Col(
                                                html.A(
                                                    "Forgot Password?", 
                                                    href="/reset-password",
                                                    style={
                                                        'color': COLORS['primary'],
                                                        'textDecoration': 'none',
                                                        'fontWeight': '500',
                                                        'fontSize': '0.9rem'
                                                    }
                                                ),
                                                className="text-end"
                                            )
                                        ], className="mb-4 align-items-center"),
                                        
                                        # Login button
                                        dbc.Button(
                                            "Sign In",
                                            id="login-button",
                                            color="primary",
                                            className="w-100 mb-4",
                                            n_clicks=0,
                                            style={
                                                'backgroundColor': COLORS['primary'],
                                                'borderColor': COLORS['primary'],
                                                'fontWeight': '500',
                                                'padding': '0.75rem',
                                                'boxShadow': COLORS['shadow'],
                                                'transition': 'all 0.2s ease'
                                            }
                                        ),
                                        
                                        # Error message
                                        html.Div(id="login-error", className="text-danger mb-3 text-center"),
                                        
                                        # Register link
                                        html.Div([
                                            "Don't have an account? ",
                                            html.A(
                                                "Register Now", 
                                                href="/register",
                                                style={
                                                    'color': COLORS['primary'],
                                                    'fontWeight': '600',
                                                    'textDecoration': 'none'
                                                }
                                            )
                                        ], className="text-center")
                                    ], style={'padding': '0.5rem 1rem'})
                                ])
                            ], style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                                'overflow': 'hidden',
                                'maxWidth': '450px',
                                'margin': '0 auto'
                            })
                        ], md=6, sm=12)
                    ])
                ], fluid=True, className="py-5")
            ]
        )
    ])

def create_register_layout():
    """Create a registration layout that matches the login page style"""
    return html.Div([
        # Background container with gradient
        html.Div(
            style={
                'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["white"]} 100%)',
                'minHeight': '100vh',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'padding': '2rem 1rem'
            },
            children=[
                # Content container
                dbc.Container([
                    dbc.Row([
                        # Left column with brand/logo (for medium screens and up)
                        dbc.Col([
                            html.Div([
                                html.I(
                                    className="fas fa-robot fa-4x mb-4",
                                    style={'color': COLORS['primary']}
                                ),
                                html.H1(
                                    "IMDA Smart Viewer",
                                    className="mb-3",
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': '700',
                                        'fontSize': '2.5rem'
                                    }
                                ),
                                html.P(
                                    "Join our platform to experience advanced document analysis powered by AI technology.",
                                    style={
                                        'color': COLORS['text'],
                                        'fontSize': '1.1rem',
                                        'opacity': '0.85',
                                        'maxWidth': '400px'
                                    }
                                ),
                                html.Hr(style={
                                    'width': '50px', 
                                    'border': f'2px solid {COLORS["secondary"]}',
                                    'opacity': '1',
                                    'margin': '2rem 0'
                                }),
                                html.P(
                                    "Create an account to start uploading and analyzing your documents today.",
                                    style={
                                        'color': COLORS['text'],
                                        'opacity': '0.7',
                                        'fontSize': '0.95rem'
                                    }
                                )
                            ], style={
                                'height': '100%',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'justifyContent': 'center',
                                'padding': '2rem'
                            })
                        ], md=6, className="d-none d-md-block"),
                        
                        # Right column with registration form
                        dbc.Col([
                            # Card with registration form
                            dbc.Card([
                                dbc.CardBody([
                                    # Header
                                    html.Div([
                                        html.H2(
                                            "Create An Account",
                                            className="text-center mb-1",
                                            style={
                                                'fontWeight': '600',
                                                'color': COLORS['text']
                                            }
                                        ),
                                        html.P(
                                            "Fill out the form to get started",
                                            className="text-center mb-4",
                                            style={
                                                'color': COLORS['text'],
                                                'opacity': '0.7'
                                            }
                                        ),
                                    ]),
                                    
                                    # Registration form
                                    html.Div([
                                        # Username field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-user"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="register-username",
                                                type="text",
                                                placeholder="Username",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-3"),
                                        
                                        # Email field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-envelope"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="register-email",
                                                type="email",
                                                placeholder="Email",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-3"),
                                        
                                        # Password field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-lock"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="register-password",
                                                type="password",
                                                placeholder="Password",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-3"),
                                        
                                        # Confirm Password field with icon
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                html.I(className="fas fa-lock"),
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'borderColor': COLORS['light_accent'],
                                                    'color': COLORS['primary']
                                                }
                                            ),
                                            dbc.Input(
                                                id="register-confirm-password",
                                                type="password",
                                                placeholder="Confirm Password",
                                                style={
                                                    'borderColor': COLORS['light_accent'],
                                                    'fontSize': '0.95rem',
                                                    'padding': '0.75rem 1rem'
                                                }
                                            )
                                        ], className="mb-4"),
                                        
                                        # Register button
                                        dbc.Button(
                                            "Create Account",
                                            id="register-button",
                                            color="primary",
                                            className="w-100 mb-4",
                                            style={
                                                'backgroundColor': COLORS['primary'],
                                                'borderColor': COLORS['primary'],
                                                'fontWeight': '500',
                                                'padding': '0.75rem',
                                                'boxShadow': COLORS['shadow'],
                                                'transition': 'all 0.2s ease'
                                            }
                                        ),
                                        
                                        # Error message
                                        html.Div(id="register-error", className="text-danger mb-3 text-center"),
                                        
                                        # Login link
                                        html.Div([
                                            "Already have an account? ",
                                            html.A(
                                                "Sign In", 
                                                href="/login",
                                                style={
                                                    'color': COLORS['primary'],
                                                    'fontWeight': '600',
                                                    'textDecoration': 'none'
                                                }
                                            )
                                        ], className="text-center")
                                    ], style={'padding': '0.5rem 1rem'})
                                ])
                            ], style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                                'overflow': 'hidden',
                                'maxWidth': '450px',
                                'margin': '0 auto'
                            })
                        ], md=6, sm=12)
                    ])
                ], fluid=True, className="py-5")
            ]
        )
    ])

def create_navbar(auth_state):
    """Create navigation bar with consistent sizing and alignment"""
    
    # Custom CSS styles
    nav_styles = {
        'navbar': {
            'boxShadow': '0 2px 4px rgba(0,0,0,0.08)',
            'background': 'linear-gradient(135deg, #ffffff 0%, #f8f7fc 100%)',
            'borderBottom': f'1px solid {COLORS["secondary"]}',
            'padding': '0.5rem 1rem',
            'minHeight': '60px',
        },
        'brand': {
            'color': COLORS['primary'],
            'fontWeight': 'bold',
            'fontSize': '1.5rem',
            'textDecoration': 'none',
            'display': 'flex',
            'alignItems': 'center',
            'gap': '10px'
        },
        'nav_link': {
            'color': COLORS['text'],
            'padding': '0.5rem 1rem',
            'borderRadius': '4px',
            'margin': '0 0.25rem',
            'transition': 'all 0.2s ease',
            'fontWeight': '500',
            'height': '38px',
            'display': 'flex',
            'alignItems': 'center',
        },
        'nav_item': {
            'display': 'flex',
            'alignItems': 'center',
        },
        'account_dropdown': {
            'height': '38px',
            'display': 'flex',
            'alignItems': 'center',
            'padding': '0.5rem 1rem',
            'borderRadius': '4px',
            'border': f'1px solid {COLORS["primary"]}',
            'color': COLORS["primary"],
            'backgroundColor': 'transparent',
            'marginLeft': '0.25rem',
        }
    }

    nav_items = []
    
    if auth_state and auth_state.get('authenticated'):
        # Admin link
        if auth_state.get('role') == 'admin':
            # Admin Dashboard link
            nav_items.append(
                dbc.NavItem(
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-shield-alt me-2"),
                            "Admin"
                        ],
                        href="/admin",
                        style=nav_styles['nav_link'],
                    ),
                    style=nav_styles['nav_item']
                )
            )
            
            # Settings link - Only for admins
            nav_items.append(
                dbc.NavItem(
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-cogs me-2"),
                            "Settings"
                        ],
                        href="/settings",
                        style=nav_styles['nav_link'],
                    ),
                    style=nav_styles['nav_item']
                )
            )
        
        # Documents link
        nav_items.append(
            dbc.NavItem(
                dbc.NavLink(
                    [
                        html.I(className="fas fa-file-alt me-2"),
                        "Documents"
                    ],
                    href="/documents",
                    style=nav_styles['nav_link'],
                ),
                style=nav_styles['nav_item']
            )
        )
        
        # Account dropdown
        nav_items.append(
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(
                        [
                            html.I(className="fas fa-user me-2"),
                            auth_state.get('username', 'User')
                        ],
                        header=True,
                    ),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem(
                        [
                            html.I(className="fas fa-sign-out-alt me-2"),
                            "Logout"
                        ],
                        id="logout-button",
                        n_clicks=0,
                        className="text-danger"
                    ),
                ],
                nav=True,
                label=html.Div(
                    [
                        html.I(className="fas fa-user-circle me-2"),
                        "Account"
                    ],
                    className="d-flex align-items-center"
                ),
                align_end=True,
                style=nav_styles['account_dropdown']
            )
        )
    else:
        nav_items.extend([
            dbc.NavItem(
                dbc.NavLink(
                    [
                        html.I(className="fas fa-sign-in-alt me-2"),
                        "Login"
                    ],
                    href="/login",
                    style=nav_styles['nav_link'],
                ),
                style=nav_styles['nav_item']
            ),
            dbc.NavItem(
                dbc.NavLink(
                    [
                        html.I(className="fas fa-user-plus me-2"),
                        "Register"
                    ],
                    href="/register",
                    style=nav_styles['nav_link'],
                ),
                style=nav_styles['nav_item']
            )
        ])

    return dbc.Navbar(
        dbc.Container([
            # Brand/logo section
            html.A(
                [
                    html.I(
                        className="fas fa-robot",
                        style={'fontSize': '14px', 'color': COLORS['primary']}
                    ),
                    html.Span(
                    "GEN AI Series",
                    style={'marginLeft': '10px',
                           'fontSize': '16px'} 
                    )
                ],
                href="/",
                style={
                    **nav_styles['brand'],
                    'marginLeft': '0',      
                    'paddingLeft': '0' },
                className="navbar-brand pe-3"
            ),
            
            # Navigation items with consistent height
            dbc.Nav(
                nav_items,
                className="ms-auto d-flex align-items-center",
                navbar=True
            ),
            ],
        fluid=True,  # Makes container take full width
        style={'paddingLeft': '1rem'} 
        ),
        color="light",
        style=nav_styles['navbar'],
        className="mb-4"
    )