# auth/admin.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara"
__version__ = "1.0"

from dash import html, dcc, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import callback_context
from sqlalchemy import func
from .models import User, UserDocument
from datetime import datetime
import traceback
import time
import json
import os
from pathlib import Path

# Define a file path for storing user tags
BASE_STORAGE_DIR = Path('storage')
USER_TAGS_FILE = BASE_STORAGE_DIR/'tags/user_default_tags.json'

def load_user_tags():
    """Load user tags from file"""
    try:
        if os.path.exists(USER_TAGS_FILE):
            with open(USER_TAGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        return {}

def save_user_tags(tags_data):
    """Save user tags to file"""
    try:
        with open(USER_TAGS_FILE, 'w') as f:
            json.dump(tags_data, f)
        return True
    except Exception as e:
        print(f"Error saving user tags: {str(e)}")
        return False

# Define the same color palette from layout.py for consistency
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

def create_admin_layout():
    """Create a redesigned admin dashboard layout that matches the login page style"""
    return html.Div(
        style={
            'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["white"]} 100%)',
            'minHeight': '100vh',
            'padding': '1.5rem',
        },
        children=[
            dbc.Container([
                # Header with title and refresh button
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(
                                className="fas fa-shield-alt fa-2x",
                                style={'color': COLORS['primary'], 'marginRight': '15px'}
                            ),
                            html.H2("Admin Dashboard", style={
                                'color': COLORS['primary'],
                                'fontWeight': '600',
                                'display': 'inline-block',
                                'verticalAlign': 'middle',
                                'margin': '0'
                            })
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ], width=8),
                    dbc.Col([
                        dbc.Button(
                            ["Refresh Data ", html.I(className="fas fa-sync")],
                            id="refresh-admin-data",
                            color="secondary",
                            className="float-end",
                            n_clicks=0,
                            style={
                                'backgroundColor': COLORS['accent'],
                                'borderColor': COLORS['accent'],
                                'boxShadow': COLORS['shadow'],
                            }
                        )
                    ], width=4, className="d-flex align-items-center justify-content-end")
                ], className="mb-4 align-items-center"),
                
                # Stats Cards Row
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-users fa-2x", style={'color': COLORS['primary']}),
                                    html.Div([
                                        html.H4("Total Users", className="card-title mb-0", style={'color': COLORS['text']}),
                                        html.H2(id="total-users-count", children="0", className="mb-0", style={'color': COLORS['primary'], 'fontWeight': '700'})
                                    ], style={'marginLeft': '15px'})
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ])
                        ], style={
                            'borderRadius': '12px',
                            'border': f'1px solid {COLORS["light_accent"]}',
                            'boxShadow': COLORS['shadow'],
                            'height': '100%'
                        }),
                        width=4, className="mb-4"
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-user-check fa-2x", style={'color': COLORS['primary']}),
                                    html.Div([
                                        html.H4("Active Users", className="card-title mb-0", style={'color': COLORS['text']}),
                                        html.H2(id="active-users-count", children="0", className="mb-0", style={'color': COLORS['primary'], 'fontWeight': '700'})
                                    ], style={'marginLeft': '15px'})
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ])
                        ], style={
                            'borderRadius': '12px',
                            'border': f'1px solid {COLORS["light_accent"]}',
                            'boxShadow': COLORS['shadow'],
                            'height': '100%'
                        }),
                        width=4, className="mb-4"
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-file-alt fa-2x", style={'color': COLORS['primary']}),
                                    html.Div([
                                        html.H4("Total Documents", className="card-title mb-0", style={'color': COLORS['text']}),
                                        html.H2(id="total-docs-count", children="0", className="mb-0", style={'color': COLORS['primary'], 'fontWeight': '700'})
                                    ], style={'marginLeft': '15px'})
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ])
                        ], style={
                            'borderRadius': '12px',
                            'border': f'1px solid {COLORS["light_accent"]}',
                            'boxShadow': COLORS['shadow'],
                            'height': '100%'
                        }),
                        width=4, className="mb-4"
                    ),
                ], className="mb-4"),
                
                # User Management Section
                dbc.Card([
                    dbc.CardBody([
                        # User Management Header and Add Button
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-user-cog", style={'color': COLORS['primary'], 'marginRight': '10px'}),
                                    html.H3("User Management", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '600',
                                        'margin': '0',
                                    })
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], width=8),
                            dbc.Col([
                                dbc.Button(
                                    ["Add User ", html.I(className="fas fa-plus")],
                                    id="add-user-button",
                                    color="primary",
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': COLORS['primary'],
                                        'borderColor': COLORS['primary'],
                                        'boxShadow': COLORS['shadow'],
                                        'fontWeight': '500',
                                        'float': 'right'
                                    }
                                )
                            ], width=4, className="d-flex align-items-center justify-content-end")
                        ], className="mb-4"),
                        
                        # Debug info (can be removed in production)
                        html.Div(id="debug-info", style={"marginBottom": "10px", "color": "gray", "fontSize": "0.8rem"}),
                        
                        # Users Table
                        dash_table.DataTable(
                            id='users-table',
                            columns=[
                                {"name": "ID", "id": "id"},
                                {"name": "Username", "id": "username"},
                                {"name": "Email", "id": "email"},
                                {"name": "Role", "id": "role"},
                                {"name": "Status", "id": "status"},
                                {"name": "Last Login", "id": "last_login"},
                                {"name": "User Tag", "id": "user_tag", "editable": True},
                                {"name": "Groups", "id": "groups"},  # New column
                                {"name": "Group Admins", "id": "group_admins"}  # New column
                            ],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'whiteSpace': 'normal',
                                'fontFamily': '"Segoe UI", Arial, sans-serif',
                                'fontSize': '14px',
                            },
                            style_header={
                                'backgroundColor': COLORS['light_accent'],
                                'fontWeight': 'bold',
                                'color': COLORS['text'],
                                'borderBottom': f'1px solid {COLORS["secondary"]}',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgba(248, 247, 252, 0.5)'
                                },
                                {
                                    'if': {'column_id': 'status', 'filter_query': '{status} eq "Active"'},
                                    'color': '#2e7d32',
                                    'fontWeight': '500'
                                },
                                {
                                    'if': {'column_id': 'status', 'filter_query': '{status} eq "Inactive"'},
                                    'color': '#c62828',
                                    'fontWeight': '500'
                                },
                                {
                                    'if': {'column_id': 'role', 'filter_query': '{role} eq "admin"'},
                                    'color': COLORS['primary'],
                                    'fontWeight': '500'
                                }
                            ],
                            row_selectable="single",
                            selected_rows=[],
                            page_size=10,
                            editable=True,  # Enable cell editing for user tags
                        ),
                        
                        # Info Text About User Tags
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", style={'color': COLORS['primary']}),
                            html.Span("You can directly edit User Tags by clicking on the cell and typing. Changes are automatically saved.")
                        ], className="mt-2 mb-3", style={"fontSize": "0.9rem", "color": COLORS['text']}),
                        
                        # Action Buttons
                        dbc.ButtonGroup([
                            dbc.Button(
                                ["Edit User ", html.I(className="fas fa-edit")],
                                id="edit-user-button",
                                color="warning",
                                className="me-2",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                }
                            ),
                            dbc.Button(
                                ["Deactivate User ", html.I(className="fas fa-user-slash")],
                                id="toggle-user-button",
                                color="danger",
                                className="me-2",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                }
                            ),
                            dbc.Button(
                                ["Delete User ", html.I(className="fas fa-trash-alt")],
                                id="delete-user-button",
                                color="danger",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                    'backgroundColor': '#dc3545',
                                    'borderColor': '#dc3545'
                                }
                            )
                        ], className="mt-3"),
                    ]),

                ], style={
                    'borderRadius': '12px',
                    'border': f'1px solid {COLORS["light_accent"]}',
                    'boxShadow': COLORS['shadow'],
                }),
                dcc.Store(id="edit-mode-store", data="add"),
                dcc.Store(id="edit-user-id-store", data=""),
                # Add/Edit User Modal
                dbc.Modal([
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            html.Div(id="user-modal-title"),
                        ),
                        close_button=True  # Add this to ensure close button works
                    ),
                    dbc.ModalBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Username", style={'fontWeight': '500', 'color': COLORS['text']}),
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
                                            type="text",
                                            id="user-username-input",
                                            placeholder="Enter username",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                                dbc.Col([
                                    dbc.Label("Email", style={'fontWeight': '500', 'color': COLORS['text']}),
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
                                            type="email",
                                            id="user-email-input",
                                            placeholder="Enter email",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Password", style={'fontWeight': '500', 'color': COLORS['text']}),
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
                                            type="password",
                                            id="user-password-input",
                                            placeholder="Enter password",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                                dbc.Col([
                                    dbc.Label("Role", style={'fontWeight': '500', 'color': COLORS['text']}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText(
                                            html.I(className="fas fa-user-tag"),
                                            style={
                                                'backgroundColor': COLORS['background'],
                                                'borderColor': COLORS['light_accent'],
                                                'color': COLORS['primary']
                                            }
                                        ),
                                        dbc.Select(
                                            id="user-role-input",
                                            options=[
                                                {"label": "User", "value": "user"},
                                                {"label": "Admin", "value": "admin"}
                                            ],
                                            value="user",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                            ], className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("User Tag", style={'fontWeight': '500', 'color': COLORS['text']}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText(
                                            html.I(className="fas fa-tag"),
                                            style={
                                                'backgroundColor': COLORS['background'],
                                                'borderColor': COLORS['light_accent'],
                                                'color': COLORS['primary']
                                            }
                                        ),
                                        dbc.Input(
                                            type="text",
                                            id="user-tag-input",
                                            placeholder="Enter user tag (optional)",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                            ], className="mt-3"),
                        ])
                    ], style={'padding': '20px'}),
                    dbc.ModalFooter([
                        dbc.Button(
                            "Close",
                            id="close-user-modal-button",  # Correct ID for the close button
                            className="me-2",
                            n_clicks=0,
                            style={
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                            }
                        ),
                        dbc.Button(
                            "Save",
                            id="save-user-button",
                            color="primary",
                            n_clicks=0,
                            style={
                                'backgroundColor': COLORS['primary'],
                                'borderColor': COLORS['primary'],
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                            }
                        ),
                    ], style={'borderTop': f'1px solid {COLORS["light_accent"]}'}),
                ], id="user-modal", is_open=False, style={'borderRadius': '12px'}),
                 dbc.Card([
                    dbc.CardBody([
                        # Groups Management Header
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-users-cog", style={'color': COLORS['primary'], 'marginRight': '10px'}),
                                    html.H3("Group Management", style={
                                        'color': COLORS['text'],
                                        'fontWeight': '600',
                                        'margin': '0',
                                    })
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], width=8),
                            dbc.Col([
                                dbc.Button(
                                    ["Add Group ", html.I(className="fas fa-plus")],
                                    id="add-group-button",
                                    color="primary",
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': COLORS['primary'],
                                        'borderColor': COLORS['primary'],
                                        'boxShadow': COLORS['shadow'],
                                        'fontWeight': '500',
                                        'float': 'right'
                                    }
                                )
                            ], width=4, className="d-flex align-items-center justify-content-end")
                        ], className="mb-4"),
                        
                        # Groups Table
                        dash_table.DataTable(
                            id='groups-table',
                            columns=[
                                {"name": "Group ID", "id": "group_id"},
                                {"name": "Group Name", "id": "name"},
                                {"name": "Description", "id": "description"},
                                {"name": "Group Admins", "id": "group_admins"},
                                {"name": "Members", "id": "members_count"},
                                {"name": "Created At", "id": "created_at"},
                            ],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'whiteSpace': 'normal',
                                'fontFamily': '"Segoe UI", Arial, sans-serif',
                                'fontSize': '14px',
                            },
                            style_header={
                                'backgroundColor': COLORS['light_accent'],
                                'fontWeight': 'bold',
                                'color': COLORS['text'],
                                'borderBottom': f'1px solid {COLORS["secondary"]}',
                            },
                            row_selectable="single",
                            selected_rows=[],
                            page_size=10,
                        ),
                        
                        # Group Action Buttons
                        dbc.ButtonGroup([
                            dbc.Button(
                                ["Edit Group ", html.I(className="fas fa-edit")],
                                id="edit-group-button",
                                color="warning",
                                className="me-2",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                }
                            ),
                            dbc.Button(
                                ["Manage Members ", html.I(className="fas fa-users-cog")],
                                id="manage-members-button",
                                color="primary",
                                className="me-2",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                }
                            ),
                            dbc.Button(
                                ["Delete Group ", html.I(className="fas fa-trash")],
                                id="delete-group-button",
                                color="danger",
                                disabled=True,
                                style={
                                    'fontWeight': '500',
                                    'boxShadow': COLORS['shadow'],
                                }
                            )
                        ], className="mt-3"),
                    ])
                ], style={
                    'borderRadius': '12px',
                    'border': f'1px solid {COLORS["light_accent"]}',
                    'boxShadow': COLORS['shadow'],
                    'marginTop': '1.5rem'
                }),
                dbc.Modal([
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle me-2", style={'color': '#dc3545'}),
                                "Confirm User Deletion"
                            ]),
                        ),
                        close_button=True
                    ),
                    dbc.ModalBody([
                        html.P("Are you sure you want to permanently delete this user?", className="mb-2"),
                        html.P("This action cannot be undone and will remove all user data including:", className="mb-3"),
                        html.Ul([
                            html.Li("User account information"),
                            html.Li("User group memberships"),
                            html.Li("User tags and custom settings")
                        ], className="mb-3"),
                        html.Div([
                            html.Strong("User to delete: "),
                            html.Span(id="delete-username-display", style={'color': '#dc3545'})
                        ], className="alert alert-danger")
                    ]),
                    dbc.ModalFooter([
                        dbc.Button(
                            "Cancel",
                            id="cancel-delete-user-button",
                            className="me-2",
                            n_clicks=0,
                            style={
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                            }
                        ),
                        dbc.Button(
                            "Delete User",
                            id="confirm-delete-user-button",
                            color="danger",
                            n_clicks=0,
                            style={
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                                'backgroundColor': '#dc3545',
                                'borderColor': '#dc3545'
                            }
                        ),
                    ]),
                ], id="delete-user-modal", is_open=False, style={'borderRadius': '12px'}),

                # Group Modal for Adding/Editing Groups
                dbc.Modal([
                    dbc.ModalHeader(
                        dbc.ModalTitle(
                            html.Div(id="group-modal-title"),
                        ),
                        close_button=True
                    ),
                    dbc.ModalBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Group Name", style={'fontWeight': '500', 'color': COLORS['text']}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText(
                                            html.I(className="fas fa-users"),
                                            style={
                                                'backgroundColor': COLORS['background'],
                                                'borderColor': COLORS['light_accent'],
                                                'color': COLORS['primary']
                                            }
                                        ),
                                        dbc.Input(
                                            type="text",
                                            id="group-name-input",
                                            placeholder="Enter group name",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ]),
                                dbc.Col([
                                    dbc.Label("Description (Optional)", style={'fontWeight': '500', 'color': COLORS['text']}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText(
                                            html.I(className="fas fa-comment-alt"),
                                            style={
                                                'backgroundColor': COLORS['background'],
                                                'borderColor': COLORS['light_accent'],
                                                'color': COLORS['primary']
                                            }
                                        ),
                                        dbc.Input(
                                            type="text",
                                            id="group-description-input",
                                            placeholder="Enter group description",
                                            style={
                                                'borderColor': COLORS['light_accent'],
                                                'fontSize': '0.95rem',
                                                'padding': '0.75rem 1rem'
                                            }
                                        )
                                    ])
                                ])
                            ], className="mb-3"),
                            
                            # Group Members Dropdown
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Group Members", style={'fontWeight': '500', 'color': COLORS['text'], 'marginBottom': '0.5rem'}),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='group-members-dropdown',
                                            multi=True,
                                            placeholder='Select group members',
                                            style={
                                                'color': COLORS['text'],
                                                'borderColor': COLORS['light_accent'],
                                                'width': '100%'
                                            },
                                            className='form-control'  # Bootstrap class to ensure consistent styling
                                        )
                                    ], style={'width': '100%'})
                                ])
                            ], className="mb-3"),
                            
                            # Group Admins Dropdown
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Group Admins", style={'fontWeight': '500', 'color': COLORS['text'], 'marginBottom': '0.5rem'}),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='group-admins-dropdown',
                                            multi=True,
                                            placeholder='Select group admins',
                                            style={
                                                'color': COLORS['text'],
                                                'borderColor': COLORS['light_accent'],
                                                'width': '100%'
                                            },
                                            className='form-control'  # Bootstrap class to ensure consistent styling
                                        )
                                    ], style={'width': '100%'})
                                ])
                            ])
                        ])
                    ], style={'padding': '20px'}),
                    dbc.ModalFooter([
                        dbc.Button(
                            "Close",
                            id="close-group-modal-button",
                            className="me-2",
                            n_clicks=0,
                            style={
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                            }
                        ),
                        dbc.Button(
                            "Update Group",
                            id="save-group-button",
                            color="primary",
                            n_clicks=0,
                            style={
                                'backgroundColor': COLORS['primary'],
                                'borderColor': COLORS['primary'],
                                'fontWeight': '500',
                                'boxShadow': COLORS['shadow'],
                            }
                        ),
                    ], style={'borderTop': f'1px solid {COLORS["light_accent"]}'}),
                ], id="group-modal", is_open=False, style={'borderRadius': '12px'}),

            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        html.Div(id="manage-members-title"),
                    ),
                    close_button=True
                ),
                dbc.ModalBody([
                    html.P("Select a member to remove from this group:"),
                    dcc.Dropdown(
                        id="group-members-select",
                        placeholder="Select a group member",
                        style={
                            'color': COLORS['text'],
                            'borderColor': COLORS['light_accent'],
                            'marginBottom': '20px'
                        }
                    ),
                    dbc.Button(
                        ["Remove from Group ", html.I(className="fas fa-user-minus")],
                        id="remove-user-from-group-button",
                        color="danger",
                        className="mt-3",
                        n_clicks=0,
                        style={
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    )
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Close",
                        id="close-manage-members-button",
                        className="me-2",
                        n_clicks=0,
                        style={
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    )
                ]),
            ], id="manage-members-modal", is_open=False, style={'borderRadius': '12px'}),

                # Feedback Messages
                dbc.Toast(
                    id="admin-feedback-toast",
                    header="Notification",
                    is_open=False,
                    dismissable=True,
                    duration=4000,
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350, "boxShadow": COLORS['shadow']},
                ),
                
                # Store components
                dcc.Store(id="selected-user-store"),
                
                # Hidden div for triggering refreshes
                html.Div(id='admin-refresh-trigger', style={'display': 'none'})
            ], fluid=True)
        ]
    )

def register_admin_callbacks(app, db_session):
    """Register all admin-related callbacks"""
    
    print("Registering admin callbacks...")

    @app.callback(
        Output("user-modal-title", "children"),
        Input("edit-mode-store", "data")
    )
    def update_modal_title(mode):
        if mode == "edit":
            return [html.I(className="fas fa-user-edit me-2", style={'color': COLORS['primary']}), "Edit User"]
        else:
            return [html.I(className="fas fa-user-plus me-2", style={'color': COLORS['primary']}), "Add New User"]
    # Debug callback to check if the button click is registered
    @app.callback(
        Output("debug-info", "children"),
        [Input("add-user-button", "n_clicks"),
         Input("refresh-admin-data", "n_clicks")]
    )
    def update_debug_info(add_clicks, refresh_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return "No buttons clicked yet"
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "add-user-button":
            return f"Add User button clicked {add_clicks} times"
        elif button_id == "refresh-admin-data":
            return f"Data refreshed at {datetime.now().strftime('%H:%M:%S')}"
        return "Button clicked"
    
    # Callback to load all admin dashboard data - triggered from multiple sources
    @app.callback(
        [
            Output("users-table", "data"),
            Output("total-users-count", "children"),
            Output("active-users-count", "children"),
            Output("total-docs-count", "children")
        ],
        [
            Input("url", "pathname"),
            Input("admin-refresh-trigger", "children"),
            Input("refresh-admin-data", "n_clicks")
        ],
        prevent_initial_call=True
    )
    def update_admin_dashboard(pathname, refresh_trigger, refresh_clicks):
        """Load all admin dashboard data with group memberships"""
        try:
            # Load user tags
            user_tags = load_user_tags()
            
            # Import GroupService to access group memberships
            from .group_management import GroupService
            group_service = GroupService()
            
            # Query users
            users = db_session.query(User).all()
            
            # Query total document count
            doc_count = db_session.query(func.count(UserDocument.id)).scalar() or 0
            
            # Prepare table data
            table_data = []
            active_count = 0
            
            for user in users:
                status = "Active" if user.is_active else "Inactive"
                if user.is_active:
                    active_count += 1
                
                last_login = user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never"
                
                # Get user tag if exists
                user_tag = user_tags.get(str(user.id), "")
                
                # Get user's group memberships
                user_groups = group_service.get_group_names_for_user(str(user.id))
                group_admin_groups = group_service.get_group_admin_groups(str(user.id))
                
                table_data.append({
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role,
                    "status": status,
                    "last_login": last_login,
                    "user_tag": user_tag,
                    "groups": ", ".join(user_groups),
                    "group_admins": ", ".join(group_admin_groups)
                })
            
            return table_data, str(len(users)), str(active_count), str(doc_count)
            
        except Exception as e:
            print(f"Error updating admin dashboard: {str(e)}")
            print(traceback.format_exc())
            return [], "Error", "Error", "Error"
    
    @app.callback(
        [
            Output("user-modal", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [
            Input("add-user-button", "n_clicks"),
            Input("close-user-modal-button", "n_clicks"),
            Input("save-user-button", "n_clicks")
        ],
        [
            State("edit-mode-store", "data"),
            State("edit-user-id-store", "value"),
            State("user-username-input", "value"),
            State("user-email-input", "value"),
            State("user-password-input", "value"),
            State("user-role-input", "value"),
            State("user-tag-input", "value"),
            State("user-modal", "is_open")
        ],
        prevent_initial_call=True
    )
    def handle_user_modal(add_clicks, close_clicks, save_clicks,
                        mode, user_id, username, email, password, role, user_tag, is_open):
        """
        Handle user modal interactions for adding and editing users
        
        Args:
        - mode: 'add' or 'edit'
        - user_id: ID of the user being edited (None for new users)
        """
        # Comprehensive logging
        print("=" * 50)
        print("USER MODAL HANDLING")
        print(f"Mode: {mode}")
        print(f"User ID: {user_id}")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Role: {role}")
        
        # Identify the triggered input
        ctx = callback_context
        if not ctx.triggered:
            print("No trigger detected")
            return False, False, "", "", ""
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        print(f"Button clicked: {button_id}")
        
        # Handle opening the modal for adding a new user
        if button_id == "add-user-button":
            print("Opening modal in add mode")
            return True, False, "", "", ""
        
        # Handle closing the modal
        elif button_id == "close-user-modal-button":
            print("Closing modal")
            return False, False, "", "", ""
        
        # Handle saving user (either new or existing)
        elif button_id == "save-user-button":
            # Input validation
            if not all([username, email, role]):
                print("Validation failed: Missing required fields")
                return is_open, True, "Error", "Please fill in all required fields", ""
            
            try:
                # Adding a new user
                if mode == "add":
                    print("Adding new user")
                    
                    # Check if user already exists
                    existing_user = db_session.query(User).filter(
                        (User.username == username) | (User.email == email)
                    ).first()
                    
                    if existing_user:
                        print("User with this username or email already exists")
                        return is_open, True, "Error", "User with this username or email already exists", ""
                    
                    # Create new user instance
                    new_user = User(
                        username=username,
                        email=email,
                        role=role,
                        is_active=True
                    )
                    
                    # Set password
                    try:
                        new_user.set_password(password)
                    except Exception as pwd_error:
                        print(f"Password setting error: {pwd_error}")
                        return is_open, True, "Error", "Invalid password", ""
                    
                    # Save to database
                    try:
                        db_session.add(new_user)
                        db_session.commit()
                        print(f"User {username} created successfully with ID: {new_user.id}")
                    except Exception as db_error:
                        print(f"Database error: {db_error}")
                        db_session.rollback()
                        return is_open, True, "Error", f"Database error: {db_error}", ""
                    
                    # Save user tag
                    if user_tag:
                        try:
                            user_tags = load_user_tags()
                            user_tags[str(new_user.id)] = user_tag
                            save_user_tags(user_tags)
                            print(f"Saved tag '{user_tag}' for user {new_user.id}")
                        except Exception as tag_error:
                            print(f"Error saving user tag: {tag_error}")
                    
                    return False, True, "Success", f"User {username} created successfully", f"refresh-{datetime.now().timestamp()}"
                
                # Editing an existing user
                elif mode == "edit":
                    # Validate user ID
                    if not user_id:
                        print("Error: No user ID provided for editing")
                        return is_open, True, "Error", "No user selected for editing", ""
                    
                    print(f"Updating existing user with ID: {user_id}")
                    
                    # Find the user
                    user = db_session.query(User).filter(User.id == user_id).first()
                    
                    if not user:
                        print(f"User with ID {user_id} not found")
                        return is_open, True, "Error", f"User with ID {user_id} not found", ""
                    
                    # Check if new username/email conflicts with other users
                    conflicting_user = db_session.query(User).filter(
                        ((User.username == username) | (User.email == email)) & 
                        (User.id != user_id)
                    ).first()
                    
                    if conflicting_user:
                        print("Username or email already in use by another user")
                        return is_open, True, "Error", "Username or email already in use", ""
                    
                    # Update user properties
                    user.username = username
                    user.email = email
                    user.role = role
                    
                    # Update password if provided
                    if password:
                        try:
                            user.set_password(password)
                        except Exception as pwd_error:
                            print(f"Password update error: {pwd_error}")
                            return is_open, True, "Error", "Invalid password", ""
                    
                    # Commit changes
                    try:
                        db_session.commit()
                        print(f"User {username} updated successfully")
                    except Exception as db_error:
                        print(f"Database error: {db_error}")
                        db_session.rollback()
                        return is_open, True, "Error", f"Database error: {db_error}", ""
                    
                    # Update user tag
                    try:
                        user_tags = load_user_tags()
                        if user_tag:
                            user_tags[str(user_id)] = user_tag
                        elif str(user_id) in user_tags:
                            del user_tags[str(user_id)]
                        save_user_tags(user_tags)
                        print(f"Updated user tag for user {user_id}")
                    except Exception as tag_error:
                        print(f"Error updating user tag: {tag_error}")
                    
                    return False, True, "Success", f"User {username} updated successfully", f"refresh-{datetime.now().timestamp()}"
                
                else:
                    print(f"Invalid mode: {mode}")
                    return is_open, True, "Error", "Invalid operation mode", ""
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                return is_open, True, "Error", f"Unexpected error: {e}", ""
        
        # Fallback
        print("No action taken")
        return is_open, False, "", "", ""
        

    @app.callback(
        [
            Output("edit-user-button", "disabled"),
            Output("toggle-user-button", "disabled"),
            Output("toggle-user-button", "children"),
            Output("delete-user-button", "disabled")  # Add this line
        ],
        [Input("users-table", "selected_rows")],
        [State("users-table", "data")],
        prevent_initial_call=True
    )
    def update_user_action_buttons(selected_rows, table_data):
        if not selected_rows or not table_data or len(selected_rows) == 0 or len(table_data) == 0:
            return True, True, ["Deactivate User ", html.I(className="fas fa-user-slash")], True  # Add True for delete button
            
        selected_row_index = selected_rows[0]
        if selected_row_index >= len(table_data):
            return True, True, ["Deactivate User ", html.I(className="fas fa-user-slash")], True  # Add True for delete button
            
        selected_user = table_data[selected_row_index]
        if selected_user["status"] == "Inactive":
            button_text = ["Activate User ", html.I(className="fas fa-user-check")]
        else:
            button_text = ["Deactivate User ", html.I(className="fas fa-user-slash")]
        
        return False, False, button_text, False
    
    # Add a new callback to save user tags when they are edited
    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True)
        ],
        [Input("users-table", "data_timestamp")],
        [State("users-table", "data")],
        prevent_initial_call=True
    )
    def save_user_tag_changes(timestamp, data):
        if not timestamp or not data:
            raise PreventUpdate
        
        try:
            # Create a dictionary of user_id to user_tag
            user_tags = {str(row["id"]): row["user_tag"] for row in data if row.get("user_tag")}
            
            # Save to file
            success = save_user_tags(user_tags)
            
            if success:
                print("User tags saved successfully")
                return True, "Success", "User tags saved successfully"
            else:
                print("Failed to save user tags")
                return True, "Error", "Failed to save user tags"
                
        except Exception as e:
            print(f"Error saving user tags: {str(e)}")
            print(traceback.format_exc())
            return True, "Error", f"Error saving user tags: {str(e)}"
    
    @app.callback(
    [
        Output("delete-user-modal", "is_open"),
        Output("delete-username-display", "children")
    ],
    [
        Input("delete-user-button", "n_clicks"),
        Input("cancel-delete-user-button", "n_clicks"),
        Input("confirm-delete-user-button", "n_clicks")
    ],
    [
        State("delete-user-modal", "is_open"),
        State("users-table", "selected_rows"),
        State("users-table", "data")
    ],
    prevent_initial_call=True
    )
    def toggle_delete_modal(delete_clicks, cancel_clicks, confirm_clicks, is_open, selected_rows, table_data):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Open modal when delete button is clicked
        if button_id == "delete-user-button" and delete_clicks:
            if not selected_rows or len(selected_rows) == 0:
                return no_update, no_update
            
            selected_row_index = selected_rows[0]
            if selected_row_index >= len(table_data):
                return no_update, no_update
            
            selected_user = table_data[selected_row_index]
            return True, selected_user["username"]
        
        # Close modal when cancel button is clicked
        elif button_id == "cancel-delete-user-button" and cancel_clicks:
            return False, no_update
        
        # Close modal after confirm button is clicked (actual deletion happens in another callback)
        elif button_id == "confirm-delete-user-button" and confirm_clicks:
            return False, no_update
        
        return no_update, no_update
    
    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [Input("toggle-user-button", "n_clicks")],
        [
            State("users-table", "selected_rows"),
            State("users-table", "data")
        ],
        prevent_initial_call=True
    )
    def toggle_user_status(n_clicks, selected_rows, table_data):
        if not n_clicks or not selected_rows or len(selected_rows) == 0:
            raise PreventUpdate
            
        try:
            selected_row_index = selected_rows[0]
            if selected_row_index >= len(table_data):
                return True, "Error", "Invalid selection", ""
                
            selected_user = table_data[selected_row_index]
            user_id = selected_user["id"]
            
            # Query the user
            user = db_session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return True, "Error", "User not found", ""
                
            # Toggle status
            user.is_active = not user.is_active
            db_session.commit()
            
            # Update the table data
            status_text = "activated" if user.is_active else "deactivated"
            
            return True, "Success", f"User {user.username} {status_text} successfully", f"toggle-{datetime.now().timestamp()}"
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error toggling user status: {error_msg}")
            print(traceback.format_exc())
            return True, "Error", f"Error updating user: {error_msg}", ""

    # Callback to handle edit button click
    @app.callback(
        [
            Output("user-modal", "is_open", allow_duplicate=True),
            Output("edit-mode-store", "data"),
            Output("edit-user-id-store", "data"),
            Output("user-username-input", "value"),
            Output("user-email-input", "value"),
            Output("user-password-input", "value"),
            Output("user-role-input", "value"),
            Output("user-tag-input", "value")
        ],
        Input("edit-user-button", "n_clicks"),
        [
            State("users-table", "selected_rows"),
            State("users-table", "data")
        ],
        prevent_initial_call=True
    )
    def handle_edit_button(n_clicks, selected_rows, table_data):
        print("=" * 50)
        print("EDIT BUTTON CLICKED")
        print(f"Number of clicks: {n_clicks}")
        print(f"Selected rows: {selected_rows}")
        print(f"Table data length: {len(table_data) if table_data else 0}")
        
        if not n_clicks or not selected_rows or len(selected_rows) == 0:
            print("No rows selected or no clicks")
            raise PreventUpdate
            
        selected_row_index = selected_rows[0]
        print(f"Selected row index: {selected_row_index}")
        
        if selected_row_index >= len(table_data):
            print("Selected row index out of range")
            raise PreventUpdate
            
        selected_user = table_data[selected_row_index]
        print(f"Selected user: {selected_user}")
        
        # Load current user_tags
        user_tags = load_user_tags()
        user_tag = user_tags.get(str(selected_user["id"]), "")
        
        print(f"Editing user: {selected_user['username']} (ID: {selected_user['id']})")
        print(f"Current tag: {user_tag}")
        
        # Make sure we return exactly 8 values corresponding to the 8 outputs
        return (
            True,               # modal is_open
            "edit",             # edit_mode_store
            str(selected_user["id"]),  # edit_user_id_store
            selected_user["username"], # username input
            selected_user["email"],    # email input
            "",                 # password input (blank for security)
            selected_user["role"],     # role input
            user_tag            # tag input
        )
    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [Input("confirm-delete-user-button", "n_clicks")],
        [
            State("users-table", "selected_rows"),
            State("users-table", "data")
        ],
        prevent_initial_call=True
    )
    def delete_user(n_clicks, selected_rows, table_data):
        if not n_clicks or not selected_rows or len(selected_rows) == 0:
            raise PreventUpdate
        
        try:
            # Get the selected user
            selected_row_index = selected_rows[0]
            if selected_row_index >= len(table_data):
                return True, "Error", "Invalid selection", ""
            
            selected_user = table_data[selected_row_index]
            user_id = selected_user["id"]
            username = selected_user["username"]
            
            print(f"Deleting user: {username} (ID: {user_id})")
            
            # 1. Remove user tags
            try:
                user_tags = load_user_tags()
                if str(user_id) in user_tags:
                    del user_tags[str(user_id)]
                    save_user_tags(user_tags)
                    print(f"Removed tags for user {user_id}")
            except Exception as tag_error:
                print(f"Error removing user tags: {tag_error}")
            
            # 2. Remove from group memberships
            try:
                # Import group service
                from .group_management import GroupService
                group_service = GroupService()
                
                # Get user's groups
                user_groups = group_service.get_user_groups(str(user_id))
                
                # Remove from all groups
                for group_id in user_groups:
                    group_service.remove_user_from_group(str(user_id), group_id)
                    print(f"Removed user {user_id} from group {group_id}")
                
                # Save changes to group files
                from .group_management import save_user_groups, save_groups
                save_user_groups(group_service.user_groups)
                save_groups(group_service.groups)
            except Exception as group_error:
                print(f"Error removing user from groups: {group_error}")
            
            # 3. Delete user documents
            try:
                # Delete UserDocument records for this user
                user_docs = db_session.query(UserDocument).filter(UserDocument.user_id == user_id).all()
                for doc in user_docs:
                    db_session.delete(doc)
                print(f"Deleted {len(user_docs)} document records for user {user_id}")
            except Exception as doc_error:
                print(f"Error deleting user documents: {doc_error}")
            
            # 4. Finally, delete the user
            user = db_session.query(User).filter(User.id == user_id).first()
            if user:
                db_session.delete(user)
                db_session.commit()
                print(f"User {username} deleted successfully")
                return True, "Success", f"User '{username}' has been permanently deleted", f"refresh-{datetime.now().timestamp()}"
            else:
                return True, "Error", f"User with ID {user_id} not found", ""
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error deleting user: {error_msg}")
            print(traceback.format_exc())
            db_session.rollback()
            return True, "Error", f"Error deleting user: {error_msg}", ""